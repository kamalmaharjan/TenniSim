# Conceptualize and optimize a tennis serve based on player attributes, racket specs, and desired serve characteristics. Then simulate the optimized serve to verify results.
from __future__ import annotations

from dataclasses import asdict

from src.backend.court import Court
from src.backend.player import Player
from src.backend.racket import Racket
from src.backend.service import Service


def run_demo() -> None:
	# These values are meant to be replaced by a UI later.
	side = "deuce"  # "deuce" | "ad"
	spin_type = "slice"  # "flat" | "slice" | "topspin"
	placement = "wide"  # "T" | "body" | "wide"
	server_start_x_m = 0.20  # baseline x position (meters); UI-controlled
	jump_height_m = 0.12
	swing_speed_mps = 35.0

	court = Court()

	racket_length_in = 27.0
	strung_weight_g = 315.0
	racket = Racket(
		length_m=racket_length_in * 0.0254,
		string_pattern="16x19",
		strung_weight_kg=strung_weight_g / 1000.0,
		head_light_balance_pts=4.0,
		swing_weight_kgcm2=320.0,
	)

	player = Player(
		height_m=1.83,
		arm_span_m=1.88,
		body_mass_kg=78.0,
		racket=racket,
	)

	service = Service(player=player, court=court)

	best = service.optimize_serve(
		swing_speed_mps=swing_speed_mps,
		side=side,
		spin_type=spin_type,
		placement=placement,
		server_start_x_m=server_start_x_m,
		jump_height_m=jump_height_m,
		log_path="data/serve_examples.jsonl",
	)

	print("=== Optimized Serve ===")
	print("side/spin/placement:", best.side, best.spin_type, best.placement)
	print("contact_point_m:", tuple(round(v, 3) for v in best.contact_point_m))
	print("toss_offset_m:", tuple(round(v, 3) for v in best.toss_offset_m))
	print("launch:", f"v0={best.launch_speed_mps:.1f} m/s", f"az={best.launch_azimuth_deg:.2f}°", f"elev={best.launch_elevation_deg:.2f}°")
	print("spin:", f"{best.spin_rpm:.0f} rpm", "axis", tuple(round(v, 2) for v in best.spin_axis_unit))
	print("landing_m:", tuple(round(v, 2) for v in best.predicted_landing_m))
	print("net_clearance_m:", round(best.net_clearance_m, 3), "margin_m:", round(best.margin_m, 3), "score:", round(best.score, 3))
	if best.racket is not None:
		print("racket:", asdict(best.racket))

	sim = service.simulate_serve(
		side=side,
		placement=placement,
		jump_height_m=jump_height_m,
		swing_speed_mps=swing_speed_mps,
		server_start_x_m=server_start_x_m,
		racket=racket,
		launch_azimuth_deg=best.launch_azimuth_deg,
		launch_elevation_deg=best.launch_elevation_deg,
		spin_rpm=best.spin_rpm,
		spin_axis_unit=best.spin_axis_unit,
	)

	print("\n=== Forward Simulation (same parameters) ===")
	print("net_crossed:", sim["net_crossed"], "net_clearance_m:", round(sim["net_clearance"], 3))
	print("landing_m:", tuple(round(v, 2) for v in sim["landing"]))
	print("t_land_s:", round(sim["t_land"], 3), "final_speed_mps:", round(sim["final_speed"], 2))


if __name__ == "__main__":
	run_demo()

