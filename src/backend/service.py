from __future__ import annotations

import json
import math
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
    BALL_AREA_M2 = float(math.pi * (BALL_RADIUS_M ** 2))

    # Aerodynamics (tuned for "reasonable" trajectories, not lab precision)
    AIR_DENSITY = 1.225
    DRAG_COEFF = 0.55

    # Spin tuning
    # - Real kick/topspin serves can be substantially higher RPM than slice.
    # - We allow higher topspin RPM than slice so the optimizer can find
    #   noticeably “dipping” trajectories.
    MAX_SPIN_RPM = 12000.0
    # Add a small sidespin component to topspin to create a kick-like sideways jump.
    # Kept constant across sides to match the existing on-screen convention.
    TOPSPIN_KICK_SIDESPIN_Z = 0.40

    # Bounce (court contact) coefficients.
    # These are tuned for visually plausible tennis bounces rather than surface-specific calibration.
    BOUNCE_RESTITUTION_Z = 0.65
    BOUNCE_FRICTION_COEFF = 0.55
    BOUNCE_SPIN_DAMP = 0.85

    # Bounce “kick” tuning: slightly higher effective friction and restitution
    # for strong topspin (x-axis spin) to mimic a more pronounced kick.
    BOUNCE_TOPSPIN_E_BOOST = 0.10
    BOUNCE_GRIP_MU_BOOST = 0.30
    BOUNCE_E_MAX = 0.82
    BOUNCE_MU_MAX = 0.95

    def _spin_axis_unit_for(self, spin_type: str) -> np.ndarray:
        """Return a unit spin axis used by the optimizer for a given spin type."""
        if spin_type == "slice":
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
        elif spin_type == "topspin":
            axis = np.array([-1.0, 0.0, float(self.TOPSPIN_KICK_SIDESPIN_Z)], dtype=float)
        else:  # flat
            axis = np.array([-1.0, 0.0, 0.0], dtype=float)

        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        return axis / n

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
            # "T" is near the center service line (x=0), but aim slightly inside.
            x = (x_min + 0.35) if side == "ad" else (x_max - 0.35)
        elif placement == "wide":
            # "Wide" is near the singles sideline, but aim slightly inside.
            x = (x_max - 0.45) if side == "ad" else (x_min + 0.45)
        else:  # body
            x = (x_min + x_max) / 2.0
        return np.array([x, y, 0.0], dtype=float)

    @staticmethod
    def _placement_margin_m(
        *,
        side: ServeSide,
        placement: ServePlacement,
        x_land: float,
        y_land: float,
        box_x_min: float,
        box_x_max: float,
        box_y_min: float,
        box_y_max: float,
    ) -> float:
        """Margin (m) to the relevant boundaries for the requested placement.

        For aggressive targets ("wide" / "T"), we intentionally *do not* penalize
        being close to the boundary line we're aiming at; otherwise the optimizer
        will systematically drift toward the middle of the box.
        """
        dx_left = x_land - box_x_min
        dx_right = box_x_max - x_land
        dy_near = y_land - box_y_min
        dy_far = box_y_max - y_land

        if placement == "wide":
            # Wide wants to hug the outside singles sideline.
            if side == "deuce":
                # deuce box is [-half_w, 0], wide is near x_min
                return float(min(dx_right, dy_near, dy_far))
            # ad box is [0, +half_w], wide is near x_max
            return float(min(dx_left, dy_near, dy_far))

        if placement == "T":
            # T wants to hug the center service line (x=0).
            if side == "deuce":
                # deuce box max boundary is x=0
                return float(min(dx_left, dy_near, dy_far))
            # ad box min boundary is x=0
            return float(min(dx_right, dy_near, dy_far))

        # body: prefer away from all lines
        return float(min(dx_left, dx_right, dy_near, dy_far))

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
        # Kept for backward compatibility; hot paths should use BALL_AREA_M2.
        return float(cls.BALL_AREA_M2)

    def _forces_accel_components(
        self,
        vx: float,
        vy: float,
        vz: float,
        wx: float,
        wy: float,
        wz: float,
    ) -> tuple[float, float, float]:
        """Acceleration from gravity + drag + Magnus (scalar fast path)."""
        speed2 = (vx * vx) + (vy * vy) + (vz * vz)
        if speed2 < 1e-18:
            return (0.0, 0.0, -9.81)
        speed = math.sqrt(speed2)

        rho = float(self.AIR_DENSITY)
        cd = float(self.DRAG_COEFF)
        area = float(self.BALL_AREA_M2)
        m = float(self.BALL_MASS_KG)

        # Quadratic drag
        drag_mag = 0.5 * rho * cd * area / m
        a_drag_x = -drag_mag * speed * vx
        a_drag_y = -drag_mag * speed * vy
        a_drag_z = -drag_mag * speed * vz

        # Magnus lift: direction is omega x v
        omega2 = (wx * wx) + (wy * wy) + (wz * wz)
        if omega2 < 1e-18:
            return (a_drag_x, a_drag_y, a_drag_z - 9.81)

        omega_mag = math.sqrt(omega2)
        S = (omega_mag * float(self.BALL_RADIUS_M)) / speed
        cl = (1.2 * S) / (1.0 + 3.0 * S)

        # Cross product omega x v
        lx = (wy * vz) - (wz * vy)
        ly = (wz * vx) - (wx * vz)
        lz = (wx * vy) - (wy * vx)
        ln2 = (lx * lx) + (ly * ly) + (lz * lz)
        if ln2 < 1e-18:
            return (a_drag_x, a_drag_y, a_drag_z - 9.81)

        inv_ln = 1.0 / math.sqrt(ln2)
        lx *= inv_ln
        ly *= inv_ln
        lz *= inv_ln

        lift_mag = 0.5 * rho * cl * area / m
        # magnitude ~ v^2
        lift_scale = lift_mag * speed2
        a_mag_x = lift_scale * lx
        a_mag_y = lift_scale * ly
        a_mag_z = lift_scale * lz

        return (
            a_drag_x + a_mag_x,
            a_drag_y + a_mag_y,
            a_drag_z + a_mag_z - 9.81,
        )

    def _forces_accel(self, vel: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Acceleration from gravity + drag + Magnus."""
        v = np.asarray(vel, dtype=float)
        w = np.asarray(omega, dtype=float)
        ax, ay, az = self._forces_accel_components(
            float(v[0]), float(v[1]), float(v[2]),
            float(w[0]), float(w[1]), float(w[2]),
        )
        return np.array([ax, ay, az], dtype=float)

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
            # Slice generally produces less total RPM than a kick/topspin serve.
            spin_mul *= 0.95
        else:  # topspin
            spin_mul *= 1.05

        spin_mul = float(np.clip(spin_mul, 0.70, 1.45))
        return speed_mul, spin_mul

    def _simulate(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        omega: np.ndarray,
        *,
        dt: float = 0.002,
        t_max: float = 3.0,
        record_path_xy: bool = False,
        path_stride_steps: int = 10,
        max_path_points: int = 260,
        simulate_bounce_after: bool = False,
        t_after_max: float = 1.2,
    ) -> dict:
        """Simulate until first ground contact (z<=0). Returns summary.

        When `record_path_xy` is True, includes a downsampled top-down path
        as `path_xy`: a list of [x_m, y_m] points.
        """
        x = float(pos0[0])
        y = float(pos0[1])
        z = float(pos0[2])
        vx = float(vel0[0])
        vy = float(vel0[1])
        vz = float(vel0[2])
        wx = float(omega[0])
        wy = float(omega[1])
        wz = float(omega[2])

        path_xy: list[list[float]] | None = None
        if record_path_xy:
            stride = int(path_stride_steps)
            stride = 1 if stride < 1 else stride
            limit = int(max_path_points)
            limit = 10 if limit < 10 else limit
            path_xy = [[float(x), float(y)]]
        else:
            stride = 1
            limit = 0

        net_y = float(self._net_y())
        net_crossed = False
        net_clearance: float | None = None

        t = 0.0
        step_i = 0

        while t < float(t_max):
            prev_x, prev_y, prev_z = x, y, z
            prev_vx, prev_vy, prev_vz = vx, vy, vz

            # Semi-implicit Euler (stable enough at small dt here)
            ax, ay, az = self._forces_accel_components(vx, vy, vz, wx, wy, wz)
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            x += vx * dt
            y += vy * dt
            z += vz * dt
            t += dt
            step_i += 1

            if path_xy is not None and (step_i % stride) == 0 and len(path_xy) < limit:
                path_xy.append([float(x), float(y)])

            # Net plane crossing interpolation
            if (not net_crossed) and (prev_y <= net_y <= y):
                net_crossed = True
                denom = (y - prev_y)
                alpha = 0.0 if abs(float(denom)) < 1e-12 else (net_y - prev_y) / denom
                x_at_net = float(prev_x + alpha * (x - prev_x))
                z_at_net = float(prev_z + alpha * (z - prev_z))
                net_clearance = z_at_net - float(self._net_height_at_x(x_at_net))

            # Ground contact (stop at first bounce)
            if z <= 0.0 and t > 0.05:
                dz = (z - prev_z)
                alpha = 0.0 if abs(float(dz)) < 1e-12 else (0.0 - prev_z) / dz
                x_land = float(prev_x + alpha * (x - prev_x))
                y_land = float(prev_y + alpha * (y - prev_y))

                # Approximate impact velocity at the contact interpolation point.
                # Linear interpolation is adequate at small dt.
                vx_imp = float(prev_vx + alpha * (vx - prev_vx))
                vy_imp = float(prev_vy + alpha * (vy - prev_vy))
                vz_imp = float(prev_vz + alpha * (vz - prev_vz))

                final_speed = math.sqrt((prev_vx * prev_vx) + (prev_vy * prev_vy) + (prev_vz * prev_vz))
                out = {
                    "t_land": float(t),
                    "landing": np.array([x_land, y_land], dtype=float),
                    "net_clearance": float(net_clearance) if net_clearance is not None else float("nan"),
                    "net_crossed": bool(net_crossed),
                    "final_speed": float(final_speed),
                }
                if path_xy is not None:
                    if len(path_xy) == 0 or (path_xy[-1][0] != float(x_land) or path_xy[-1][1] != float(y_land)):
                        path_xy.append([float(x_land), float(y_land)])
                    out["path_xy"] = path_xy

                # Optional: simulate the post-bounce segment (2nd flight) with a simple
                # impulse-based bounce (restitution + Coulomb friction + spin coupling).
                if simulate_bounce_after:
                    m = float(self.BALL_MASS_KG)
                    R = float(self.BALL_RADIUS_M)
                    # Solid sphere inertia.
                    I = (2.0 / 5.0) * m * (R * R)

                    e = float(self.BOUNCE_RESTITUTION_Z)
                    mu = float(self.BOUNCE_FRICTION_COEFF)
                    spin_damp = float(self.BOUNCE_SPIN_DAMP)

                    # Spin-aware “grip” model:
                    # - Higher spin increases effective friction slightly.
                    # - Strong topspin (about x) also nudges restitution upward,
                    #   producing a more kick-like bounce.
                    omega_mag = math.sqrt((wx * wx) + (wy * wy) + (wz * wz))
                    vtan_mag = math.hypot(vx_imp, vy_imp)
                    # Dimensionless spin ratio ~ (omega*R)/v.
                    spin_ratio = (omega_mag * R) / max(1.0, float(vtan_mag))
                    grip = max(0.0, min(1.0, float(spin_ratio / 0.55)))
                    mu = float(min(float(self.BOUNCE_MU_MAX), mu * (1.0 + float(self.BOUNCE_GRIP_MU_BOOST) * grip)))

                    topspin_frac = 0.0 if omega_mag < 1e-9 else abs(float(wx)) / float(omega_mag)
                    e = float(
                        min(
                            float(self.BOUNCE_E_MAX),
                            e + float(self.BOUNCE_TOPSPIN_E_BOOST) * grip * float(topspin_frac),
                        )
                    )

                    # Normal impulse (z): reflect with restitution.
                    vz_in = float(vz_imp)
                    if vz_in > 0.0:
                        vz_in = -abs(vz_in)
                    Jn = -(1.0 + e) * m * vz_in  # >= 0
                    vz_out = -e * vz_in

                    # Tangential friction impulse with spin coupling.
                    # Relative tangential velocity at the contact point (ground frame):
                    # v_rel = v_t + (omega x r)_t, where r = (0,0,-R).
                    vrel_x = float(vx_imp + (wy * R))
                    vrel_y = float(vy_imp - (wx * R))
                    vrel_mag = math.hypot(vrel_x, vrel_y)

                    Jtx = 0.0
                    Jty = 0.0
                    if vrel_mag > 1e-9 and Jn > 0.0:
                        ux = vrel_x / vrel_mag
                        uy = vrel_y / vrel_mag

                        # Impulse required to bring slip to ~0 (no-slip) for a sphere:
                        # v_rel' = v_rel + J_t*(1/m + R^2/I)
                        # => J_req = -v_rel / (1/m + R^2/I)
                        denom = (1.0 / m) + ((R * R) / I)
                        J_req = vrel_mag / denom

                        # Coulomb friction limit
                        J_max = mu * Jn
                        J_t_mag = min(J_req, J_max)

                        # Friction impulse opposes slip
                        Jtx = -J_t_mag * ux
                        Jty = -J_t_mag * uy

                    # Apply impulses to linear velocity
                    vx_out = float(vx_imp + (Jtx / m))
                    vy_out = float(vy_imp + (Jty / m))

                    # Apply impulses to angular velocity (about center): omega' = omega + (r x J)/I
                    # r = (0,0,-R), J = (Jtx, Jty, 0) => r x J = (R*Jty, -R*Jtx, 0)
                    wx_out = float((wx + (R * Jty) / I) * spin_damp)
                    wy_out = float((wy + (-R * Jtx) / I) * spin_damp)
                    wz_out = float(wz * spin_damp)

                    out["bounce_e"] = float(e)
                    out["bounce_mu"] = float(mu)
                    out["bounce_vz_out"] = float(vz_out)

                    x2 = float(x_land)
                    y2 = float(y_land)
                    z2 = 0.001
                    vx2 = float(vx_out)
                    vy2 = float(vy_out)
                    vz2 = float(vz_out)

                    # Use the post-bounce spin for post-bounce flight curvature.
                    wx2 = float(wx_out)
                    wy2 = float(wy_out)
                    wz2 = float(wz_out)

                    path_xy_post: list[list[float]] | None = None
                    if record_path_xy:
                        path_xy_post = [[float(x2), float(y2)]]
                    t2 = 0.0
                    step2 = 0

                    while t2 < float(t_after_max):
                        px, py, pz = x2, y2, z2
                        pvx, pvy, pvz = vx2, vy2, vz2

                        ax2, ay2, az2 = self._forces_accel_components(vx2, vy2, vz2, wx2, wy2, wz2)
                        vx2 += ax2 * dt
                        vy2 += ay2 * dt
                        vz2 += az2 * dt
                        x2 += vx2 * dt
                        y2 += vy2 * dt
                        z2 += vz2 * dt
                        t2 += dt
                        step2 += 1

                        if path_xy_post is not None and (step2 % stride) == 0 and len(path_xy_post) < limit:
                            path_xy_post.append([float(x2), float(y2)])

                        if z2 <= 0.0 and t2 > 0.05:
                            dz2 = (z2 - pz)
                            alpha2 = 0.0 if abs(float(dz2)) < 1e-12 else (0.0 - pz) / dz2
                            x_land2 = float(px + alpha2 * (x2 - px))
                            y_land2 = float(py + alpha2 * (y2 - py))
                            out["t_land2"] = float(t + t2)
                            out["landing2"] = np.array([x_land2, y_land2], dtype=float)
                            if path_xy_post is not None:
                                if len(path_xy_post) == 0 or (path_xy_post[-1][0] != float(x_land2) or path_xy_post[-1][1] != float(y_land2)):
                                    path_xy_post.append([float(x_land2), float(y_land2)])
                                out["path_xy_post"] = path_xy_post
                            break
                    else:
                        # Time ran out before 2nd contact.
                        out["t_land2"] = float("nan")
                        out["landing2"] = np.array([float("nan"), float("nan")], dtype=float)
                        if path_xy_post is not None:
                            out["path_xy_post"] = path_xy_post
                return out

        final_speed = math.sqrt((vx * vx) + (vy * vy) + (vz * vz))
        out = {
            "t_land": float("nan"),
            "landing": np.array([float("nan"), float("nan")], dtype=float),
            "net_clearance": float("nan"),
            "net_crossed": bool(net_crossed),
            "final_speed": float(final_speed),
        }
        if path_xy is not None:
            out["path_xy"] = path_xy
        return out

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
        return_path_xy: bool = False,
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

        sim = self._simulate(
            contact,
            vel0,
            omega,
            record_path_xy=bool(return_path_xy),
            simulate_bounce_after=bool(return_path_xy),
        )
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
        launch_speed_base = float(max(10.0, (speed_factor * speed_mul) * float(swing_speed_mps)))

        aggressive_grid = placement in {"wide", "T"}

        # Spin search grid (RPM)
        # We center the grid around a *typical* RPM based on swing speed and spin type,
        # while still allowing variation (more for aggressive placements).
        # Key constraint we want: topspin serves tend to produce ~2× the RPM of slice.
        swing = float(swing_speed_mps)
        if spin_type == "flat":
            spin_rpm_typ = 20.0 * swing
            facs = np.array([0.0, 0.7, 1.0, 1.25], dtype=float)
        elif spin_type == "slice":
            # Typical slice RPM scale.
            spin_rpm_typ = 110.0 * swing
            facs = (
                np.array([0.85, 1.00, 1.15], dtype=float)
                if aggressive_grid
                else np.array([0.90, 1.00, 1.10], dtype=float)
            )
        else:  # topspin/kick
            # Typical topspin/kick RPM scale (~2× slice by construction).
            spin_rpm_typ = 220.0 * swing
            facs = (
                np.array([0.80, 0.95, 1.10], dtype=float)
                if aggressive_grid
                else np.array([0.85, 1.00, 1.10], dtype=float)
            )

        spin_rpm_grid = np.clip((spin_rpm_typ * spin_mul) * facs, 0.0, float(self.MAX_SPIN_RPM))

        # Placement-specific aggressiveness: for wide/T (and explicit click targets),
        # prioritize the chosen spot over conservative "stay centered" margins.
        eff_target_weight = float(target_weight)
        eff_margin_weight = float(margin_weight)
        if placement in {"wide", "T"}:
            eff_target_weight *= 1.6
            eff_margin_weight *= 0.75
        if target_xy_m is not None:
            eff_target_weight *= 1.25
            eff_margin_weight *= 0.85

        # Small speed grid: lets the optimizer find better depth when aiming wide/T.
        # For non-aggressive placements, keep speed fixed for performance.
        if aggressive_grid:
            speed_grid = np.array([1.02, 1.14], dtype=float)
        else:
            speed_grid = np.array([1.00], dtype=float)

        # Search angles.
        # We center azimuth around the geometric line from contact -> target.
        # Elevation includes more negative (downward) angles so the ball can land
        # inside the service box at high launch speeds.
        to_target_xy = target[:2] - contact[:2]
        az0_deg = float(np.rad2deg(np.arctan2(to_target_xy[0], to_target_xy[1])))
        az_span = 16.0 if aggressive_grid else 14.0
        az_grid = np.linspace(az0_deg - az_span, az0_deg + az_span, 25 if aggressive_grid else 21)
        az_grid = np.clip(az_grid, -35.0, 35.0)

        if spin_type == "flat":
            elev_grid = np.linspace(-18.0, 8.0, 21)
        elif spin_type == "slice":
            elev_grid = np.linspace(-16.0, 10.0, 21)
        else:  # topspin/kick
            elev_grid = np.linspace(-12.0, 16.0, 23)

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

                for speed_fac in speed_grid:
                    launch_speed = float(launch_speed_base * float(speed_fac))
                    vel0 = launch_speed * dir_vec

                    for spin_rpm in spin_rpm_grid:
                        omega_mag = float(spin_rpm) * 2.0 * np.pi / 60.0
                        if omega_mag < 1e-9:
                            omega = np.zeros(3, dtype=float)
                            spin_axis = np.array([0.0, 0.0, 0.0], dtype=float)
                        else:
                            # Spin axis selection (heuristic):
                            # - Slice: omega ~ +z (lateral Magnus; keep same on-screen direction across sides).
                            # - Topspin: mostly -x (downward Magnus) plus some +z for a kick-like jump.
                            # - Flat: small topspin by default (also -x) so nonzero RPM actually has an effect.
                            spin_axis = self._spin_axis_unit_for(spin_type)
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

                        # Margin to lines. For "wide"/"T", don't penalize being close
                        # to the intended boundary line, otherwise results cluster mid-box.
                        margin = self._placement_margin_m(
                            side=side,
                            placement=placement,
                            x_land=x_land,
                            y_land=y_land,
                            box_x_min=float(box_x_min),
                            box_x_max=float(box_x_max),
                            box_y_min=float(box_y_min),
                            box_y_max=float(box_y_max),
                        )

                        # Distance to desired target point in the box
                        target_err = float(np.linalg.norm(np.array([x_land, y_land, 0.0]) - target))

                        # Score: prefer speed but heavily penalize missing the chosen spot.
                        score = (
                            speed_weight * (launch_speed / 70.0)
                            + eff_margin_weight * margin
                            - eff_target_weight * target_err
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
                                launch_speed_mps=float(launch_speed),
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

        # Local refinement: for aggressive targets, do a small fine-grained search
        # around the best coarse solution. This improves "hug the line" serves
        # without paying the cost of a dense global grid.
        if best is not None and aggressive_grid:
            az_fine = np.linspace(float(best.launch_azimuth_deg) - 6.0, float(best.launch_azimuth_deg) + 6.0, 21)
            elev_fine = np.linspace(float(best.launch_elevation_deg) - 3.0, float(best.launch_elevation_deg) + 3.0, 17)
            az_fine = np.clip(az_fine, -35.0, 35.0)

            for elev_deg in elev_fine:
                elev = np.deg2rad(float(elev_deg))
                for az_deg in az_fine:
                    az = np.deg2rad(float(az_deg))

                    dir_vec = np.array(
                        [
                            np.sin(az) * np.cos(elev),
                            np.cos(az) * np.cos(elev),
                            np.sin(elev),
                        ],
                        dtype=float,
                    )

                    for speed_fac in speed_grid:
                        launch_speed = float(launch_speed_base * float(speed_fac))
                        vel0 = launch_speed * dir_vec

                        for spin_rpm in spin_rpm_grid:
                            omega_mag = float(spin_rpm) * 2.0 * np.pi / 60.0
                            if omega_mag < 1e-9:
                                omega = np.zeros(3, dtype=float)
                                spin_axis = np.array([0.0, 0.0, 0.0], dtype=float)
                            else:
                                spin_axis = self._spin_axis_unit_for(spin_type)
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

                            margin = self._placement_margin_m(
                                side=side,
                                placement=placement,
                                x_land=x_land,
                                y_land=y_land,
                                box_x_min=float(box_x_min),
                                box_x_max=float(box_x_max),
                                box_y_min=float(box_y_min),
                                box_y_max=float(box_y_max),
                            )

                            target_err = float(np.linalg.norm(np.array([x_land, y_land, 0.0]) - target))

                            score = (
                                speed_weight * (launch_speed / 70.0)
                                + eff_margin_weight * margin
                                - eff_target_weight * target_err
                            )
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
                                    launch_speed_mps=float(launch_speed),
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