from dataclasses import dataclass
from typing import Literal

from src.backend.racket import Racket

ServeSide = Literal["deuce", "ad"]
ServeSpin = Literal["flat", "slice", "topspin"]
ServePlacement = Literal["T", "body", "wide"]

@dataclass(frozen=True)
class ServeParams:
    side: ServeSide
    spin_type: ServeSpin
    placement: ServePlacement

    contact_point_m: tuple[float, float, float]  # (x, y, z)
    toss_offset_m: tuple[float, float, float]  # relative to contact (x, y, z)

    launch_speed_mps: float
    launch_azimuth_deg: float  # left/right aim relative to +y axis
    launch_elevation_deg: float

    spin_rpm: float
    spin_axis_unit: tuple[float, float, float]  # direction of angular velocity vector

    predicted_landing_m: tuple[float, float]
    net_clearance_m: float
    margin_m: float

    score: float

    # Optional metadata (useful for UI and training logs)
    racket: Racket | None = None