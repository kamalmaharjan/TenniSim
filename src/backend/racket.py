from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


StringPattern = Literal["16x19", "18x20", "16x20", "18x19"]


@dataclass(frozen=True)
class Racket:
		"""Racket parameters that affect serve biomechanics/spin potential.

		Units:
		- length_m: meters
		- strung_weight_kg: kilograms (static weight, strung)
		- head_light_balance_pts: "points" head-light (+) / head-heavy (-)
			(1 point = 1/8 inch). Positive means more head-light.
		- swing_weight_kgcm2: conventional swingweight units (kg·cm²)
		"""

		# 27" standard length
		length_m: float = 27.0 * 0.0254

		# Common patterns
		string_pattern: StringPattern = "16x19"

		# Typical adult racket: ~300–330g strung
		strung_weight_kg: float = 0.315

		# Typical balance: a few points head-light
		head_light_balance_pts: float = 4.0

		# Typical swingweight: ~310–330
		swing_weight_kgcm2: float = 320.0
