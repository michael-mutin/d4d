"""A parametric union-cuboid shape grammar.

An object is a string of ``A`` symbols with one row of twelve parameters per
symbol. The parameters are ``(cx, cy, cz, sx, sy, sz, alpha, beta, gamma,
r, g, b)``: center, half-size, Euler rotation angles in radians, and RGB.

Run it with ``uv run python tests/uc.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from d4d import ExtraMetrics, Grammar, ListCollection, ListSpec, StepContext

####################
# The Object
####################

@dataclass(frozen=True)
class CubSet:
    """A sequence of cuboids represented by ``A`` symbols.

    ``params[i]`` belongs to ``program[i]`` and contains
    ``(cx, cy, cz, sx, sy, sz, alpha, beta, gamma, r, g, b)``.
    """

    program: str
    params: torch.Tensor

    def __str__(self) -> str:
        return " ".join(
            "A(" + ", ".join(f"{value:.2f}" for value in row) + ")"
            for row in self.params.detach().tolist()
        )


def render_cubset(
    cubset: CubSet,
    *,
    ax: Any | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    elev: float | None = 25.0,
    azim: float | None = -60.0,
) -> Any:
    """Render all cuboids in ``cubset`` as a Matplotlib 3D figure.

    The three rotation values are interpreted as intrinsic X, Y, and Z
    rotations in radians.  Pass ``show=True`` to open an interactive window;
    its normal Matplotlib mouse controls can be used to orbit the camera.
    ``elev`` and ``azim`` set the initial camera elevation and azimuth.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt

    if cubset.params.ndim != 2 or cubset.params.shape[1] != 12:
        raise ValueError("CubSet.params must have shape (number_of_cuboids, 12)")
    if len(cubset.program) != cubset.params.shape[0]:
        raise ValueError("CubSet.program and CubSet.params must have the same length")

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    all_vertices: list[np.ndarray] = []
    for row in cubset.params.detach().cpu().numpy():
        center = row[0:3]
        half_size = row[3:6]
        alpha, beta, gamma = row[6:9]
        color = np.clip(row[9:12], 0.0, 1.0)

        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
        cos_beta, sin_beta = np.cos(beta), np.sin(beta)
        cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)
        rotate_x = np.array(
            [[1, 0, 0], [0, cos_alpha, -sin_alpha], [0, sin_alpha, cos_alpha]]
        )
        rotate_y = np.array(
            [[cos_beta, 0, sin_beta], [0, 1, 0], [-sin_beta, 0, cos_beta]]
        )
        rotate_z = np.array(
            [[cos_gamma, -sin_gamma, 0], [sin_gamma, cos_gamma, 0], [0, 0, 1]]
        )
        rotation = rotate_z @ rotate_y @ rotate_x

        corners = np.array(
            [
                [x, y, z]
                for x in (-half_size[0], half_size[0])
                for y in (-half_size[1], half_size[1])
                for z in (-half_size[2], half_size[2])
            ]
        )
        vertices = corners @ rotation.T + center
        all_vertices.append(vertices)
        faces = [
            [vertices[index] for index in (0, 1, 3, 2)],
            [vertices[index] for index in (4, 6, 7, 5)],
            [vertices[index] for index in (0, 4, 5, 1)],
            [vertices[index] for index in (2, 3, 7, 6)],
            [vertices[index] for index in (0, 2, 6, 4)],
            [vertices[index] for index in (1, 5, 7, 3)],
        ]
        ax.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=color,
                edgecolors=(0.12, 0.12, 0.12),
                linewidths=0.45,
            )
        )

    if all_vertices:
        bounds = np.concatenate(all_vertices)
        minimum = bounds.min(axis=0)
        maximum = bounds.max(axis=0)
        center = (minimum + maximum) / 2
        radius = max((maximum - minimum).max() / 2, 1e-6) * 1.08
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))

    ax.set_axis_off()
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return ax.figure

####################
# The Rewrites
####################

@dataclass(frozen=True)
class SplitX:
    """Split the cuboid at position ``i`` along the x-axis at relative position ``t``."""

    i: int
    t: float


@dataclass(frozen=True)
class SplitY:
    """Split the cuboid at position ``i`` along the y-axis at relative position ``t``."""

    i: int
    t: float


@dataclass(frozen=True)
class SplitZ:
    """Split the cuboid at position ``i`` along the z-axis at relative position ``t``."""

    i: int
    t: float


@dataclass(frozen=True)
class AddCub:
    """Add a cuboid at position ``(x, y, z)`` with size ``s``."""

    x: float
    y: float
    z: float
    s: float


@dataclass(frozen=True)
class AddHoleXY:
    """Cut a hole at ``(x, y)`` with radius ``s`` in the cuboid at position ``i``."""

    i: int
    x: float
    y: float
    s: float


@dataclass(frozen=True)
class AddHoleXZ:
    """Cut a hole at ``(x, z)`` with radius ``s`` in the cuboid at position ``i``."""

    i: int
    x: float
    z: float
    s: float


@dataclass(frozen=True)
class AddHoleYZ:
    """Cut a hole at ``(y, z)`` with radius ``s`` in the cuboid at position ``i``."""

    i: int
    y: float
    z: float
    s: float


@dataclass(frozen=True)
class RemoveRect:
    """Remove the cuboid at position ``i``."""

    i: int


@dataclass(frozen=True)
class Merge:
    """Merge the cuboids at positions ``i`` and ``j`` into a new cuboid."""

    i: int
    j: int


Rewrite = SplitX | SplitY | SplitZ | AddCub | AddHoleXY | AddHoleXZ | AddHoleYZ | RemoveRect | Merge

####################
# The Grammar
####################

# Grammar[TObject, TCollection, TRewrite, TState]
class CubGrammar(Grammar[CubSet, ListCollection[CubSet], Rewrite, None]):
    """Placeholder grammar for parametric union-cuboid objects."""

    list_spec = ListSpec(
        params_of=lambda o: [o.params],
        with_params=lambda o, ts: replace(o, params=ts[0]),
        names=("params",),
    )

    def initial(self) -> CubSet:
        raise NotImplementedError

    def propose(self, obj: CubSet, budget: int) -> list[Rewrite]:
        raise NotImplementedError

    def apply(self, obj: CubSet, rewrite: Rewrite) -> CubSet:
        raise NotImplementedError

    def conflicts(self, a: Rewrite, b: Rewrite) -> bool:
        raise NotImplementedError

    def apply_all(self, base: CubSet, rewrites: Sequence[Rewrite]) -> CubSet:
        raise NotImplementedError

    def loss(
        self, batch: ListCollection[CubSet], ctx: StepContext, state: None
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        raise NotImplementedError

    def simplicity(self, batch: ListCollection[CubSet], ctx: StepContext) -> list[float]:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        raise NotImplementedError

