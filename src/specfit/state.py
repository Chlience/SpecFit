
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class ExpertKey:

    layer_id: int
    expert_id: int

    def __post_init__(self) -> None:
        if self.layer_id < 0 or self.expert_id < 0:
            raise ValueError("layer_id and expert_id must be non-negative")
