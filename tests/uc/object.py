"""Objects used by the union-cuboid grammar."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CubSet:
    """A sequence of cuboids represented by ``A`` symbols."""

    program: str
    params: torch.Tensor

    def __str__(self) -> str:
        return " ".join(
            "A(" + ", ".join(f"{value:.2f}" for value in row) + ")"
            for row in self.params.detach().tolist()
        )
