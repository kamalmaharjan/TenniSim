from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
from typing import Any, TypedDict

from flask import Flask, jsonify, render_template, request, send_from_directory

try:
	from flask_cors import CORS  # type: ignore
except Exception:  # noqa: BLE001
	CORS = None  # type: ignore[assignment]

from src.backend.court import Court
from src.backend.player import Player
from src.backend.racket import Racket
from src.backend.service import Service


ROOT_DIR = Path(__file__).resolve().parent
# ROOT_DIR = <project>/src/frontend
# parents[0] = <project>/src
# parents[1] = <project>
PROJECT_DIR = ROOT_DIR.parents[1]
IMGS_DIR = PROJECT_DIR / "imgs"


class OptimizeRequest(TypedDict, total=False):
	side: str
	spin_type: str
	placement: str
	server_start_x_m: float
	jump_height_m: float
	swing_speed_mps: float

	height_m: float
	arm_span_m: float
	body_mass_kg: float

	racket_length_in: float
	string_pattern: str
	strung_weight_g: float
	head_light_balance_pts: float
	swing_weight_kgcm2: float

	# Optional custom target (meters in backend coord system)
	target_x_m: float
	target_y_m: float


def create_app() -> Flask:
	app = Flask(
		__name__,
		template_folder=str(ROOT_DIR / "templates"),
	)
	
	@app.after_request
	def add_cors_headers(resp):
		# Allow a static frontend (e.g., GitHub Pages) to call the JSON API.
		# NOTE: You can lock this down to a specific origin later if desired.
		resp.headers.setdefault("Access-Control-Allow-Origin", "*")
		resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
		resp.headers.setdefault("Access-Control-Max-Age", "86400")
		return resp

	@app.get("/")
	def index_get():
		defaults: dict[str, Any] = {
			"side": "deuce",
			"spin_type": "slice",
			"placement": "wide",
			"server_start_x_m": 0.20,
			"jump_height_m": 0.12,
			"swing_speed_mps": 35.0,
			"height_m": 1.83,
			"arm_span_m": 1.88,
			"body_mass_kg": 78.0,
			"racket_length_in": 27.0,
			"string_pattern": "18x20",
			"strung_weight_g": 345.0,
			"head_light_balance_pts": 7.0,
			"swing_weight_kgcm2": 330.0,

			# Optional: click-to-set target
			"target_x_m": "",
			"target_y_m": "",
		}
		return render_template("index.html", form=defaults, result=None, error=None)

	@app.get("/imgs/<path:filename>")
	def imgs(filename: str):
		return send_from_directory(str(IMGS_DIR), filename)

	@app.post("/")
	def index_post():
		def f(name: str, default: float) -> float:
			raw = request.form.get(name, "")
			return float(raw) if raw != "" else float(default)

		def f_opt(name: str) -> float | None:
			raw = request.form.get(name, "")
			return None if raw == "" else float(raw)

		server_start_x_m = f("server_start_x_m", 0.20)
		form = {
			"side": request.form.get("side", ""),
			"spin_type": request.form.get("spin_type", "slice"),
			"placement": request.form.get("placement", ""),
			"server_start_x_m": server_start_x_m,
			"jump_height_m": f("jump_height_m", 0.12),
			"swing_speed_mps": f("swing_speed_mps", 35.0),
			"height_m": f("height_m", 1.83),
			"arm_span_m": f("arm_span_m", 1.88),
			"body_mass_kg": f("body_mass_kg", 78.0),
			"racket_length_in": f("racket_length_in", 27.0),
			"string_pattern": request.form.get("string_pattern", "18x20"),
			"strung_weight_g": f("strung_weight_g", 345.0),
			"head_light_balance_pts": f("head_light_balance_pts", 7.0),
			"swing_weight_kgcm2": f("swing_weight_kgcm2", 330.0),

			"target_x_m": request.form.get("target_x_m", ""),
			"target_y_m": request.form.get("target_y_m", ""),
		}

		# Also keep parsed optional target for the optimizer.
		target_x_m = f_opt("target_x_m")
		target_y_m = f_opt("target_y_m")
		if target_x_m is not None:
			form["target_x_m"] = target_x_m
		if target_y_m is not None:
			form["target_y_m"] = target_y_m

		# Infer side/placement if not provided (or invalid)
		court = Court()
		side = str(form.get("side") or "").strip().lower()
		if side not in {"deuce", "ad"}:
			side = "ad" if float(server_start_x_m) < 0.0 else "deuce"
		form["side"] = side

		placement = str(form.get("placement") or "").strip()
		if placement not in {"wide", "body", "T"}:
			placement = "wide"
			if target_x_m is not None:
				half_w = float(court.width) / 2.0
				if side == "deuce":
					u = (float(target_x_m) - (-half_w)) / (half_w - (-half_w))
					if u < 0.33:
						placement = "wide"
					elif u > 0.66:
						placement = "T"
					else:
						placement = "body"
				else:
					u = (float(target_x_m) - 0.0) / (half_w - 0.0)
					if u < 0.33:
						placement = "T"
					elif u > 0.66:
						placement = "wide"
					else:
						placement = "body"
		form["placement"] = placement

		try:
			result = _run_optimizer(form)  # type: ignore[arg-type]
			return render_template("index.html", form=form, result=result, error=None)
		except Exception as e:  # noqa: BLE001
			return render_template("index.html", form=form, result=None, error=str(e))

	@app.route("/api/optimize", methods=["POST", "OPTIONS"])
	def api_optimize():
		if request.method == "OPTIONS":
			# Preflight request for CORS
			return ("", 204)
		try:
			payload = request.get_json(force=True, silent=False) or {}
		except Exception as e:  # noqa: BLE001
			return jsonify({"error": f"Invalid JSON: {e}"}), 400
		try:
			result = _run_optimizer(payload)
		except Exception as e:  # noqa: BLE001
			return jsonify({"error": str(e)}), 500
		return jsonify(result)

	return app


def _run_optimizer(payload: OptimizeRequest) -> dict[str, Any]:
	court = Court()

	server_start_x_m = float(payload.get("server_start_x_m", 0.20))
	side = str(payload.get("side") or "").strip().lower()
	if side not in {"deuce", "ad"}:
		side = "ad" if server_start_x_m < 0.0 else "deuce"

	placement = str(payload.get("placement") or "").strip()

	if "racket_length_in" in payload:
		length_m = float(payload.get("racket_length_in", 27.0)) * 0.0254
	else:
		length_m = float(payload.get("racket_length_m", 27.0 * 0.0254))  # type: ignore[typeddict-item]

	if "strung_weight_g" in payload:
		strung_weight_kg = float(payload.get("strung_weight_g", 345.0)) / 1000.0
	else:
		strung_weight_kg = float(payload.get("strung_weight_kg", 0.345))  # type: ignore[typeddict-item]

	racket = Racket(
		length_m=length_m,
		string_pattern=str(payload.get("string_pattern", "18x20")),
		strung_weight_kg=strung_weight_kg,
		head_light_balance_pts=float(payload.get("head_light_balance_pts", 7.0)),
		swing_weight_kgcm2=float(payload.get("swing_weight_kgcm2", 330.0)),
	)
	player = Player(
		height_m=float(payload.get("height_m", 1.70)),
		arm_span_m=float(payload.get("arm_span_m", 1.80)),
		body_mass_kg=float(payload.get("body_mass_kg", 95.0)),
		racket=racket,
	)
	service = Service(player=player, court=court)

	target_xy_m: tuple[float, float] | None = None
	tx_raw = payload.get("target_x_m")
	ty_raw = payload.get("target_y_m")
	if tx_raw is not None and ty_raw is not None:
		try:
			tx = float(tx_raw)  # type: ignore[arg-type]
			ty = float(ty_raw)  # type: ignore[arg-type]
			target_xy_m = (tx, ty)
		except (TypeError, ValueError):
			target_xy_m = None

	# If placement wasn't provided, infer from target_x when available.
	if placement not in {"wide", "body", "T"}:
		placement = "wide"
		if target_xy_m is not None:
			tx = float(target_xy_m[0])
			half_w = float(court.width) / 2.0
			if side == "deuce":
				u = (tx - (-half_w)) / (half_w - (-half_w))
				if u < 0.33:
					placement = "wide"
				elif u > 0.66:
					placement = "T"
				else:
					placement = "body"
			else:
				u = (tx - 0.0) / (half_w - 0.0)
				if u < 0.33:
					placement = "T"
				elif u > 0.66:
					placement = "wide"
				else:
					placement = "body"

	swing_speed_mps = float(payload.get("swing_speed_mps", 42.0))
	jump_height_m = float(payload.get("jump_height_m", 0.5))

	best = service.optimize_serve(
		swing_speed_mps=swing_speed_mps,
		side=side,
		spin_type=str(payload.get("spin_type", "slice")),
		placement=placement,
		target_xy_m=target_xy_m,
		server_start_x_m=server_start_x_m,
		jump_height_m=jump_height_m,
	)

	# Use the optimizer's chosen launch speed when doing forward sims.
	launch_speed_factor = float(best.launch_speed_mps) / max(1e-9, float(swing_speed_mps))

	sim = service.simulate_serve(
		side=side,
		placement=placement,
		jump_height_m=jump_height_m,
		swing_speed_mps=swing_speed_mps,
		server_start_x_m=server_start_x_m,
		racket=racket,
		launch_azimuth_deg=float(best.launch_azimuth_deg),
		launch_elevation_deg=float(best.launch_elevation_deg),
		spin_rpm=float(best.spin_rpm),
		spin_axis_unit=best.spin_axis_unit,
		launch_speed_factor=launch_speed_factor,
		return_path_xy=True,
	)

	# Variance: run a small Monte Carlo ensemble around the optimized serve.
	# This intentionally introduces slight run-to-run variation while keeping quality.
	rng = random.Random()
	n_samples = 16
	sigma_az_deg = 0.60
	sigma_elev_deg = 0.60
	sigma_speed_frac = 0.015
	sigma_spin_frac = 0.03

	sim_samples: list[dict[str, Any]] = []
	for _ in range(n_samples):
		az = float(best.launch_azimuth_deg) + rng.gauss(0.0, sigma_az_deg)
		elev = float(best.launch_elevation_deg) + rng.gauss(0.0, sigma_elev_deg)
		speed_fac = float(launch_speed_factor) * max(0.70, 1.0 + rng.gauss(0.0, sigma_speed_frac))
		spin_rpm = max(0.0, float(best.spin_rpm) * max(0.70, 1.0 + rng.gauss(0.0, sigma_spin_frac)))

		s = service.simulate_serve(
			side=side,
			placement=placement,
			jump_height_m=jump_height_m,
			swing_speed_mps=swing_speed_mps,
			server_start_x_m=server_start_x_m,
			racket=racket,
			launch_azimuth_deg=az,
			launch_elevation_deg=elev,
			spin_rpm=spin_rpm,
			spin_axis_unit=best.spin_axis_unit,
			launch_speed_factor=speed_fac,
			return_path_xy=True,
		)
		sim_samples.append(
			{
				"landing": [float(s["landing"][0]), float(s["landing"][1])],
				"landing2": (
					None
					if s.get("landing2") is None
					else [float(s["landing2"][0]), float(s["landing2"][1])]
				),
				"net_crossed": bool(s.get("net_crossed")),
				"net_clearance": float(s.get("net_clearance", float("nan"))),
				"t_land": float(s.get("t_land", float("nan"))),
				"t_land2": float(s.get("t_land2", float("nan"))),
				"path_xy": s.get("path_xy"),
				"path_xy_post": s.get("path_xy_post"),
			}
		)

	# Summary stats for the UI.
	box_x_min, box_x_max, box_y_min, box_y_max = service._service_box_bounds(side)
	xy = [
		(tuple(s["landing"]) if isinstance(s.get("landing"), list) and len(s["landing"]) == 2 else None)
		for s in sim_samples
	]
	xy2 = [(float(x), float(y)) for p in xy if p is not None for x, y in [p]]
	if xy2:
		xs = [p[0] for p in xy2]
		ys = [p[1] for p in xy2]
		xm = sum(xs) / len(xs)
		ym = sum(ys) / len(ys)
		xv = sum((x - xm) ** 2 for x in xs) / len(xs)
		yv = sum((y - ym) ** 2 for y in ys) / len(ys)
		sx = xv ** 0.5
		sy = yv ** 0.5
		p_in_box = sum(1 for x, y in xy2 if (box_x_min <= x <= box_x_max and box_y_min <= y <= box_y_max)) / len(xy2)
	else:
		xm = ym = sx = sy = float("nan")
		p_in_box = float("nan")

	sim_stats = {
		"n": int(n_samples),
		"sigma": {
			"az_deg": float(sigma_az_deg),
			"elev_deg": float(sigma_elev_deg),
			"speed_frac": float(sigma_speed_frac),
			"spin_frac": float(sigma_spin_frac),
		},
		"landing_mean": [float(xm), float(ym)],
		"landing_std": [float(sx), float(sy)],
		"p_in_box": float(p_in_box),
	}

	return {
		"input": dict(payload),
		"best": asdict(best),
		"sim": {
			**{k: v for k, v in sim.items() if k not in {"landing", "landing2"}},
			"landing": [float(sim["landing"][0]), float(sim["landing"][1])],
			"landing2": (
				None
				if sim.get("landing2") is None
				else [float(sim["landing2"][0]), float(sim["landing2"][1])]
			),
		},
		"sim_samples": sim_samples,
		"sim_stats": sim_stats,
		"best_user_units": {
			"racket_length_in": float(best.racket.length_m) / 0.0254,
			"strung_weight_g": float(best.racket.strung_weight_kg) * 1000.0,
		},
		"target": {
			"requested_xy_m": None if target_xy_m is None else [float(target_xy_m[0]), float(target_xy_m[1])],
			"box_bounds_m": list(service._service_box_bounds(side)),
		},
	}


app = create_app()

if CORS is not None:
	CORS(
		app,
		resources={r"/api/*": {"origins": ["https://kamalmaharjan.github.io"]}},
		methods=["GET", "POST", "OPTIONS"],
		allow_headers=["Content-Type"],
	)