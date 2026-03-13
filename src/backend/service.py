from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.backend.court import Court
from src.backend.player import Player
from src.backend.racket import Racket, StringPattern
from src.backend.service_params import ServeParams

import numpy as np

if TYPE_CHECKING:
    from src.backend.service_params import ServePlacement, ServeSide, ServeSpin


@dataclass(frozen=True)
class ServeExample:
    """One (inputs -> outputs) record for later model training."""

    # Player / equipment
    height_m: float
    arm_span_m: float
    body_mass_kg: float | None
    racket: Racket

    # Serve setup
    side: ServeSide
    spin_type: ServeSpin
    placement: ServePlacement
    server_start_x_m: float
    jump_height_m: float
    swing_speed_mps: float
    toss_offset_m: tuple[float, float, float]

    # Chosen launch
    launch_speed_mps: float
    launch_azimuth_deg: float
    launch_elevation_deg: float
    spin_rpm: float
    spin_axis_unit: tuple[float, float, float]

    # Outcome
    predicted_landing_m: tuple[float, float]
    net_clearance_m: float
    margin_m: float
    score: float

class Service:
    """Serve model + optimizer.

    Coordinate system (meters):
    - +y points from server baseline toward the net/opponent.
    - +x points to the server's right when facing the net.
    - +z is up.

    This uses a simplified physics model (gravity + quadratic drag + Magnus lift).
    The goal is not perfect realism, but a usable optimizer that responds
    sensibly to height/arm-span, swing speed, spin type, and deuce/ad side.
    """

    # Tennis ball constants
    BALL_RADIUS_M = 0.0335
    BALL_MASS_KG = 0.057

    # Aerodynamics (tuned for "reasonable" trajectories, not lab precision)
    AIR_DENSITY = 1.225
    DRAG_COEFF = 0.55

    def __init__(self, player: Player | None = None, court: Court | None = None):
        self.player = player
        self.court = court
        self.service_params: ServeParams | None = None

    def _get_racket(self, racket: Racket | None) -> Racket:
        if racket is not None:
            return racket
        if self.player is not None and getattr(self.player, "racket", None) is not None:
            return self.player.racket
        return Racket()

    # -------------------------
    # Geometry / targets
    # -------------------------
    def _net_y(self) -> float:
        return float(self.court.length) / 2.0

    def _service_box_bounds(self, side: ServeSide) -> tuple[float, float, float, float]:
        """Returns (x_min, x_max, y_min, y_max) for the *correct* opponent service box."""
        net_y = self._net_y()
        y_min = net_y
        y_max = net_y + float(self.court.service_line_distance)

        half_w = float(self.court.width) / 2.0
        if side == "deuce":
            # Must land in opponent's deuce box: x in [-half_w, 0]
            return (-half_w, 0.0, y_min, y_max)
        # ad: x in [0, +half_w]
        return (0.0, half_w, y_min, y_max)

    def _target_point(self, side: ServeSide, placement: ServePlacement) -> np.ndarray:
        x_min, x_max, y_min, y_max = self._service_box_bounds(side)
        # Aim a bit inside the service line for speed + margin.
        y = y_max - 0.8
        if placement == "T":
            x = (x_max - 0.35) if side == "ad" else (x_min + 0.35)
        elif placement == "wide":
            x = (x_max - 0.45) if side == "ad" else (x_min + 0.45)
        else:  # body
            x = (x_min + x_max) / 2.0
        return np.array([x, y, 0.0], dtype=float)

    def _default_server_start_x(self, side: ServeSide) -> float:
        """Default baseline x near the center mark (UI-friendly)."""
        # Legal serve is from the correct half of the baseline; we pick a spot close
        # to the center mark so the UI can easily move left/right from there.
        offset = 0.60
        return offset if side == "deuce" else -offset

    def _server_start_x(self, side: ServeSide, server_start_x_m: float | None) -> float:
        half_w = float(self.court.width) / 2.0
        x = float(self._default_server_start_x(side) if server_start_x_m is None else server_start_x_m)
        # Clamp to singles court width. (UI can still place anywhere within the lines.)
        return float(np.clip(x, -half_w, half_w))

    def _net_height_at_x(self, x: float) -> float:
        """Approximate net height varying linearly from center to posts."""
        half_w = float(self.court.width) / 2.0
        t = min(1.0, abs(float(x)) / half_w)
        return float(self.court.net_height_center) + t * (float(self.court.net_height_posts) - float(self.court.net_height_center))

    # -------------------------
    # Player reach / contact / toss
    # -------------------------
    def estimate_contact_height_m(
        self,
        *,
        jump_height_m: float = 0.12,
        reach_factor: float = 0.95,
        racket: Racket | None = None,
    ) -> float:
        """Estimate serve contact height.

        Uses height + arm span + racket length. The parameters are deliberately
        exposed because biomechanics vary a lot between players.
        """
        height_m = float(self.player.height_m)
        arm_span_m = float(self.player.arm_span_m)
        racket_used = self._get_racket(racket)

        # Rough anthropometric model:
        shoulder_height = 0.82 * height_m
        arm_plus_hand = 0.46 * arm_span_m + 0.08
        racket_effective = 0.90 * float(racket_used.length_m)

        return reach_factor * (shoulder_height + arm_plus_hand + racket_effective + float(jump_height_m))

    def _recommended_toss_offset(
        self,
        side: ServeSide,
        spin_type: ServeSpin,
        placement: ServePlacement,
    ) -> np.ndarray:
        # Relative to contact point (x, y, z). +y is into court.
        # Flat/slice: toss slightly in front. Topspin/kick: more above/behind.
        forward = 0.45 if spin_type in ("flat", "slice") else 0.20
        up = 0.55 if spin_type == "flat" else (0.65 if spin_type == "slice" else 0.85)

        # Lateral toss helps slice/placements.
        lateral = 0.0
        if spin_type == "slice":
            # Curve away from receiver: deuce -> left (negative x), ad -> right (positive x)
            lateral = -0.18 if side == "deuce" else 0.18
        if placement == "wide":
            lateral += (-0.10 if side == "deuce" else 0.10)
        elif placement == "T":
            lateral += (0.05 if side == "deuce" else -0.05)

        return np.array([lateral, forward, up], dtype=float)

    # -------------------------
    # Physics model
    # -------------------------
    @classmethod
    def _ball_area(cls) -> float:
        return float(np.pi * (cls.BALL_RADIUS_M ** 2))

    def _forces_accel(self, vel: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Acceleration from gravity + drag + Magnus."""
        v = np.asarray(vel, dtype=float)
        speed = float(np.linalg.norm(v))
        if speed < 1e-9:
            return np.array([0.0, 0.0, -9.81], dtype=float)

        area = self._ball_area()
        rho = self.AIR_DENSITY
        m = self.BALL_MASS_KG

        # Quadratic drag
        drag_mag = 0.5 * rho * self.DRAG_COEFF * area / m
        a_drag = -drag_mag * speed * v

        # Magnus lift: direction is omega x v
        omega_vec = np.asarray(omega, dtype=float)
        omega_mag = float(np.linalg.norm(omega_vec))
        if omega_mag < 1e-9:
            a_magnus = np.zeros(3, dtype=float)
        else:
            # Spin parameter S = omega*r / v
            S = (omega_mag * self.BALL_RADIUS_M) / speed
            # Simple saturating lift coefficient curve
            cl = (1.2 * S) / (1.0 + 3.0 * S)

            lift_dir = np.cross(omega_vec, v)
            lift_norm = float(np.linalg.norm(lift_dir))
            if lift_norm < 1e-9:
                a_magnus = np.zeros(3, dtype=float)
            else:
                lift_dir = lift_dir / lift_norm
                lift_mag = 0.5 * rho * cl * area / m
                a_magnus = lift_mag * (speed ** 2) * lift_dir

        return a_drag + a_magnus + np.array([0.0, 0.0, -9.81], dtype=float)

    def _racket_speed_spin_multipliers(self, racket: Racket, *, spin_type: str) -> tuple[float, float]:
        """Heuristic multipliers (speed_mul, spin_mul) from racket properties.

        This intentionally models *tendencies*:
        - Higher swingweight / higher static weight: harder to accelerate (lower speed), but more stability.
        - More head-light: easier to accelerate (higher speed/spin potential).
        - String pattern: open patterns help spin.
        """
        # Baselines
        sw0 = 320.0
        w0 = 0.315
        hl0 = 4.0

        sw = float(racket.swing_weight_kgcm2)
        w = float(racket.strung_weight_kg)
        hl = float(racket.head_light_balance_pts)

        speed_mul = 1.0
        speed_mul *= 1.0 - 0.10 * ((sw - sw0) / 40.0)
        speed_mul *= 1.0 - 0.06 * ((w - w0) / 0.03)
        speed_mul *= 1.0 + 0.012 * (hl - hl0)
        speed_mul = float(np.clip(speed_mul, 0.75, 1.10))

        # Pattern -> spin
        pattern_bonus: dict[StringPattern, float] = {
            "16x19": 1.08,
            "16x20": 1.03,
            "18x19": 0.99,
            "18x20": 0.94,
        }
        spin_mul = float(pattern_bonus.get(racket.string_pattern, 1.0))

        # Head-light and swingweight both impact spin generation differently.
        spin_mul *= 1.0 + 0.010 * (hl - hl0)
        spin_mul *= 1.0 + 0.030 * ((sw - sw0) / 40.0)

        # Spin-type nuance
        if spin_type == "flat":
            spin_mul *= 0.6
        elif spin_type == "slice":
            spin_mul *= 0.95
        else:  # topspin
            spin_mul *= 1.05

        spin_mul = float(np.clip(spin_mul, 0.70, 1.35))
        return speed_mul, spin_mul

    def _simulate(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        omega: np.ndarray,
        *,
        dt: float = 0.002,
        t_max: float = 3.0,
    ) -> dict:
        """Simulate until first ground contact (z<=0). Returns summary."""
        pos = np.asarray(pos0, dtype=float).copy()
        vel = np.asarray(vel0, dtype=float).copy()
        omega = np.asarray(omega, dtype=float)

        net_y = self._net_y()
        net_crossed = False
        net_clearance = None

        t = 0.0
        prev_pos = pos.copy()
        prev_vel = vel.copy()

        while t < t_max:
            prev_pos = pos.copy()
            prev_vel = vel.copy()

            # Semi-implicit Euler (stable enough at small dt here)
            acc = self._forces_accel(vel, omega)
            vel = vel + acc * dt
            pos = pos + vel * dt
            t += dt

            # Net plane crossing interpolation
            if (not net_crossed) and (prev_pos[1] <= net_y <= pos[1]):
                net_crossed = True
                # Interpolate by y
                denom = (pos[1] - prev_pos[1])
                alpha = 0.0 if abs(float(denom)) < 1e-9 else (net_y - prev_pos[1]) / denom
                x_at_net = float(prev_pos[0] + alpha * (pos[0] - prev_pos[0]))
                z_at_net = float(prev_pos[2] + alpha * (pos[2] - prev_pos[2]))
                net_clearance = z_at_net - self._net_height_at_x(x_at_net)

            # Ground contact (on opponent side only matters, but we stop at first bounce)
            if pos[2] <= 0.0 and t > 0.05:
                # interpolate to z=0 for landing (linear)
                dz = pos[2] - prev_pos[2]
                alpha = 0.0 if abs(float(dz)) < 1e-9 else (0.0 - prev_pos[2]) / dz
                x_land = float(prev_pos[0] + alpha * (pos[0] - prev_pos[0]))
                y_land = float(prev_pos[1] + alpha * (pos[1] - prev_pos[1]))
                return {
                    "t_land": float(t),
                    "landing": np.array([x_land, y_land], dtype=float),
                    "net_clearance": float(net_clearance) if net_clearance is not None else float("nan"),
                    "net_crossed": bool(net_crossed),
                    "final_speed": float(np.linalg.norm(prev_vel)),
                }

        return {
            "t_land": float("nan"),
            "landing": np.array([float("nan"), float("nan")], dtype=float),
            "net_clearance": float("nan"),
            "net_crossed": bool(net_crossed),
            "final_speed": float(np.linalg.norm(vel)),
        }

    def simulate_serve(
        self,
        *,
        side: ServeSide,
        placement: ServePlacement,
        jump_height_m: float,
        swing_speed_mps: float,
        server_start_x_m: float | None = None,
        racket: Racket | None = None,
        launch_azimuth_deg: float,
        launch_elevation_deg: float,
        spin_rpm: float,
        spin_axis_unit: tuple[float, float, float],
        launch_speed_factor: float | None = None,
    ) -> dict:
        """Run one forward simulation from UI-selected parameters.

        This is the method to call from an interactive frontend when the user
        adjusts baseline position, angles, toss, etc.
        """
        if self.court is None:
            raise ValueError("Service.simulate_serve requires a Court instance")
        if self.player is None:
            raise ValueError("Service.simulate_serve requires a Player instance")

        racket_used = self._get_racket(racket)

        contact_z = self.estimate_contact_height_m(jump_height_m=jump_height_m, racket=racket_used)
        contact = np.array([self._server_start_x(side, server_start_x_m), 0.0, contact_z], dtype=float)

        # Ball speed from swing speed.
        speed_factor = 1.75 if launch_speed_factor is None else float(launch_speed_factor)
        launch_speed = float(max(10.0, speed_factor * float(swing_speed_mps)))

        elev = np.deg2rad(float(launch_elevation_deg))
        az = np.deg2rad(float(launch_azimuth_deg))
        dir_vec = np.array(
            [
                np.sin(az) * np.cos(elev),
                np.cos(az) * np.cos(elev),
                np.sin(elev),
            ],
            dtype=float,
        )
        vel0 = launch_speed * dir_vec

        omega_mag = float(spin_rpm) * 2.0 * np.pi / 60.0
        axis = np.asarray(spin_axis_unit, dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        axis = axis / axis_norm if axis_norm > 1e-9 else np.zeros(3, dtype=float)
        omega = omega_mag * axis

        sim = self._simulate(contact, vel0, omega)
        sim["contact_point_m"] = (float(contact[0]), float(contact[1]), float(contact[2]))
        sim["launch_speed_mps"] = float(launch_speed)
        sim["side"] = side
        sim["placement"] = placement
        sim["racket"] = asdict(racket_used)
        return sim

    @staticmethod
    def append_example_jsonl(path: str | Path, example: ServeExample) -> None:
        """Append one training row to a JSONL file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

    # -------------------------
    # Optimization
    # -------------------------
    def optimize_serve(
        self,
        *,
        swing_speed_mps: float,
        side: ServeSide | None = None,
        spin_type: ServeSpin | None = None,
        placement: ServePlacement | None = None,
        target_xy_m: tuple[float, float] | None = None,
        server_start_x_m: float | None = None,
        toss_offset_m: tuple[float, float, float] | None = None,
        jump_height_m: float = 0.12,
        racket: Racket | None = None,
        speed_weight: float = 1.0,
        margin_weight: float = 18.0,
        target_weight: float = 10.0,
        min_net_clearance_m: float = 0.05,
        log_path: str | Path | None = None,
    ) -> ServeParams:
        """Search launch + spin parameters for a serve that lands in the right box.

        `swing_speed_mps` is racket-head speed at impact (m/s).
        Returns a single best ServeParams for the specified side/spin/placement.
        """
        if self.court is None:
            raise ValueError("Service.optimize_serve requires a Court instance")
        if self.player is None:
            raise ValueError("Service.optimize_serve requires a Player instance")

        # Backwards-compat: if not provided, pull from self.service_params.
        if side is None or spin_type is None or placement is None:
            if self.service_params is None:
                raise ValueError("Provide side/spin_type/placement or set self.service_params")
            side = self.service_params.side
            spin_type = self.service_params.spin_type
            placement = self.service_params.placement

        racket_used = self._get_racket(racket)

        contact_z = self.estimate_contact_height_m(jump_height_m=jump_height_m, racket=racket_used)
        contact = np.array([self._server_start_x(side, server_start_x_m), 0.0, contact_z], dtype=float)

        box_x_min, box_x_max, box_y_min, box_y_max = self._service_box_bounds(side)
        if target_xy_m is None:
            target = self._target_point(side, placement)
        else:
            tx, ty = target_xy_m
            tx = float(np.clip(float(tx), float(box_x_min), float(box_x_max)))
            ty = float(np.clip(float(ty), float(box_y_min), float(box_y_max)))
            target = np.array([tx, ty, 0.0], dtype=float)

        # Impact efficiency: ball speed is some multiple of racket speed.
        # Flat converts best to ball speed; spin serves trade speed for rotation.
        if spin_type == "flat":
            speed_factor = 1.85
        elif spin_type == "slice":
            speed_factor = 1.70
        else:  # topspin/kick
            speed_factor = 1.58

        speed_mul, spin_mul = self._racket_speed_spin_multipliers(racket_used, spin_type=spin_type)
        launch_speed = float(max(10.0, (speed_factor * speed_mul) * float(swing_speed_mps)))

        # Spin ranges (scaled by racket pattern/balance/swingweight heuristic)
        if spin_type == "flat":
            base = np.array([0.0, 400.0, 900.0])
        elif spin_type == "slice":
            base = np.array([1400.0, 2400.0, 3400.0, 4400.0])
        else:
            base = np.array([2400.0, 3800.0, 5200.0, 6800.0])
        spin_rpm_grid = np.clip(base * spin_mul, 0.0, 9000.0)

        # Search angles.
        # We center azimuth around the geometric line from contact -> target.
        # Elevation includes more negative (downward) angles so the ball can land
        # inside the service box at high launch speeds.
        to_target_xy = target[:2] - contact[:2]
        az0_deg = float(np.rad2deg(np.arctan2(to_target_xy[0], to_target_xy[1])))
        az_span = 14.0
        az_grid = np.linspace(az0_deg - az_span, az0_deg + az_span, 29)
        az_grid = np.clip(az_grid, -35.0, 35.0)

        if spin_type == "flat":
            elev_grid = np.linspace(-18.0, 8.0, 27)
        elif spin_type == "slice":
            elev_grid = np.linspace(-16.0, 10.0, 27)
        else:  # topspin/kick
            elev_grid = np.linspace(-12.0, 16.0, 29)

        best: ServeParams | None = None
        best_score = -float("inf")

        for elev_deg in elev_grid:
            elev = np.deg2rad(elev_deg)
            for az_deg in az_grid:
                az = np.deg2rad(az_deg)

                # Convert angles to direction: azimuth around z, elevation from horizontal.
                dir_vec = np.array(
                    [
                        np.sin(az) * np.cos(elev),
                        np.cos(az) * np.cos(elev),
                        np.sin(elev),
                    ],
                    dtype=float,
                )
                vel0 = launch_speed * dir_vec

                for spin_rpm in spin_rpm_grid:
                    omega_mag = float(spin_rpm) * 2.0 * np.pi / 60.0
                    if omega_mag < 1e-9:
                        omega = np.zeros(3, dtype=float)
                        spin_axis = np.array([0.0, 0.0, 0.0], dtype=float)
                    else:
                        # Spin axis selection (handedness-free heuristic):
                        # - Slice: omega ~ +z (gives lateral Magnus). Sign depends on side.
                        # - Topspin: omega ~ -x (gives downward Magnus).
                        if spin_type == "slice":
                            spin_axis = np.array([0.0, 0.0, -1.0 if side == "deuce" else 1.0], dtype=float)
                        elif spin_type == "topspin":
                            spin_axis = np.array([-1.0, 0.0, 0.0], dtype=float)
                        else:
                            spin_axis = np.array([0.0, 0.0, 0.0], dtype=float)
                        omega = omega_mag * spin_axis

                    sim = self._simulate(contact, vel0, omega)
                    if not sim["net_crossed"]:
                        continue
                    if not np.isfinite(sim["net_clearance"]) or sim["net_clearance"] < min_net_clearance_m:
                        continue

                    landing = sim["landing"]
                    x_land = float(landing[0])
                    y_land = float(landing[1])

                    in_box = (box_x_min <= x_land <= box_x_max) and (box_y_min <= y_land <= box_y_max)
                    if not in_box:
                        continue

                    # Margin to lines (minimum distance to any boundary)
                    margin = min(
                        x_land - box_x_min,
                        box_x_max - x_land,
                        y_land - box_y_min,
                        box_y_max - y_land,
                    )

                    # Distance to desired target point in the box
                    target_err = float(np.linalg.norm(np.array([x_land, y_land, 0.0]) - target))

                    # Score: prefer speed but heavily penalize missing the chosen spot.
                    score = (
                        speed_weight * (launch_speed / 70.0)
                        + margin_weight * margin
                        - target_weight * target_err
                    )
                    # Encourage safe net clearance slightly (but not too much)
                    score += 2.0 * float(sim["net_clearance"])

                    if score > best_score:
                        toss = np.array(toss_offset_m, dtype=float) if toss_offset_m is not None else self._recommended_toss_offset(side, spin_type, placement)
                        best_score = score
                        best = ServeParams(
                            side=side,
                            spin_type=spin_type,
                            placement=placement,
                            contact_point_m=(float(contact[0]), float(contact[1]), float(contact[2])),
                            toss_offset_m=(float(toss[0]), float(toss[1]), float(toss[2])),
                            launch_speed_mps=launch_speed,
                            launch_azimuth_deg=float(az_deg),
                            launch_elevation_deg=float(elev_deg),
                            spin_rpm=float(spin_rpm),
                            spin_axis_unit=(float(spin_axis[0]), float(spin_axis[1]), float(spin_axis[2])),
                            predicted_landing_m=(x_land, y_land),
                            net_clearance_m=float(sim["net_clearance"]),
                            margin_m=float(margin),
                            score=float(score),
                            racket=racket_used,
                        )

        if best is None:
            raise RuntimeError(
                "No valid serve found. Try lowering `min_net_clearance_m`, "
                "reducing `target_weight`, or increasing swing speed."
            )

        if log_path is not None:
            example = ServeExample(
                height_m=float(self.player.height_m),
                arm_span_m=float(self.player.arm_span_m),
                body_mass_kg=None if self.player.body_mass_kg is None else float(self.player.body_mass_kg),
                racket=racket_used,
                side=best.side,
                spin_type=best.spin_type,
                placement=best.placement,
                server_start_x_m=float(contact[0]),
                jump_height_m=float(jump_height_m),
                swing_speed_mps=float(swing_speed_mps),
                toss_offset_m=best.toss_offset_m,
                launch_speed_mps=float(best.launch_speed_mps),
                launch_azimuth_deg=float(best.launch_azimuth_deg),
                launch_elevation_deg=float(best.launch_elevation_deg),
                spin_rpm=float(best.spin_rpm),
                spin_axis_unit=best.spin_axis_unit,
                predicted_landing_m=best.predicted_landing_m,
                net_clearance_m=float(best.net_clearance_m),
                margin_m=float(best.margin_m),
                score=float(best.score),
            )
            self.append_example_jsonl(log_path, example)

        return best

    # Backwards-compatible stub (kept because your file had it).
    def perform_serve(self):
        raise NotImplementedError(
            "Use `optimize_serve(...)` to compute serve parameters based on physics and constraints."
        )