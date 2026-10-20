"""Matplotlib rendering for union-cuboid objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .uc_object import CubSet


def render_cubset(
    cubset: CubSet,
    *,
    ax: Any | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    elev: float | None = 25.0,
    azim: float | None = -60.0,
) -> Any:
    """Render all cuboids in ``cubset`` as a Matplotlib 3D figure."""
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
        rotate_x = np.array([[1, 0, 0], [0, cos_alpha, -sin_alpha], [0, sin_alpha, cos_alpha]])
        rotate_y = np.array([[cos_beta, 0, sin_beta], [0, 1, 0], [-sin_beta, 0, cos_beta]])
        rotate_z = np.array([[cos_gamma, -sin_gamma, 0], [sin_gamma, cos_gamma, 0], [0, 0, 1]])
        rotation = rotate_z @ rotate_y @ rotate_x

        corners = np.array(
            [[x, y, z] for x in (-half_size[0], half_size[0])
             for y in (-half_size[1], half_size[1])
             for z in (-half_size[2], half_size[2])]
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
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=(0.12, 0.12, 0.12), linewidths=0.45))

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


def render_cubset_bw(
    cubset: CubSet,
    *,
    ax: Any | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    elev: float | None = 25.0,
    azim: float | None = -60.0,
) -> Any:
    """Render the cubset as a black object on a white background."""
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
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")

    all_vertices: list[np.ndarray] = []
    for row in cubset.params.detach().cpu().numpy():
        center = row[0:3]
        half_size = row[3:6]
        alpha, beta, gamma = row[6:9]

        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
        cos_beta, sin_beta = np.cos(beta), np.sin(beta)
        cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)
        rotate_x = np.array([[1, 0, 0], [0, cos_alpha, -sin_alpha], [0, sin_alpha, cos_alpha]])
        rotate_y = np.array([[cos_beta, 0, sin_beta], [0, 1, 0], [-sin_beta, 0, cos_beta]])
        rotate_z = np.array([[cos_gamma, -sin_gamma, 0], [sin_gamma, cos_gamma, 0], [0, 0, 1]])
        rotation = rotate_z @ rotate_y @ rotate_x

        corners = np.array(
            [[x, y, z] for x in (-half_size[0], half_size[0])
             for y in (-half_size[1], half_size[1])
             for z in (-half_size[2], half_size[2])]
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
        ax.add_collection3d(Poly3DCollection(faces, facecolors="black", edgecolors="black", linewidths=0.45))

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
        ax.figure.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    return ax.figure
