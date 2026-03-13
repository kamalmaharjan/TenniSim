from dataclasses import dataclass, field

from src.backend.racket import Racket

@dataclass(frozen=True)
class Player:
    height_m: float
    arm_span_m: float
    body_mass_kg: float | None = None
    racket: Racket = field(default_factory=Racket)