"""Objects used by the union-cuboid grammar."""

from __future__ import annotations

from dataclasses import dataclass

import torch

PARAMETER_DIM = 12
SCENE_CENTER = (0.0, 1.25, 0.0)
SCENE_HALF_EXTENT = 2.0
ROTATION_SCALE = torch.pi


def denormalize_cuboid_params(params: torch.Tensor) -> torch.Tensor:
    """Convert normalized cuboid parameters to the renderer's world space."""
    if params.ndim != 2 or params.shape[1] != PARAMETER_DIM:
        raise ValueError(f"CubSet.params must have shape (number_of_cuboids, {PARAMETER_DIM})")

    world = params.clone()
    center = params.new_tensor(SCENE_CENTER)
    world[:, :3] = center + params[:, :3] * SCENE_HALF_EXTENT
    world[:, 3:6] = params[:, 3:6] * SCENE_HALF_EXTENT
    world[:, 6:9] = params[:, 6:9] * ROTATION_SCALE
    return world


@dataclass(frozen=True)
class CubSet:
    """A sequence of cuboids represented by ``A`` symbols.

    Parameters use normalized coordinates: positions and half-sizes use the
    scene half-extent as one unit, rotations are radians divided by pi, and
    colors are in the range [0, 1].
    """

    program: str
    params: torch.Tensor

    def __str__(self) -> str:
        return " ".join(
            "A(" + ", ".join(f"{value:.2f}" for value in row) + ")"
            for row in self.params.detach().tolist()
        )
