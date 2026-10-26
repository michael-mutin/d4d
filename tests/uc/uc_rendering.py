"""Matplotlib rendering for union-cuboid objects.

The view uses Matplotlib's ``vertical_axis="y"`` so its elevation and azimuth
match PyTorch3D's Y-up camera convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .uc_object import CubSet

DEFAULT_IMAGE_BOUNDS = (-1.5, 1.5, -0.75, 3.25, -1.5, 1.5)
MATPLOTLIB_VERTICAL_AXIS = "y"


def _figure_rgb(figure: Any) -> torch.Tensor:
    """Read a rendered Matplotlib figure as an ``(H, W, 3)`` float tensor."""
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    return torch.from_numpy(rgba).float() / 255.0


def render_cubset_rgb_image(
    cubset: CubSet,
    camera_pose: tuple[float | None, float | None] = (25.0, -60.0),
) -> torch.Tensor:
    """Render a cubset with :func:`render_cubset` and return an RGB image tensor."""
    figure = render_cubset(cubset, elev=camera_pose[0], azim=camera_pose[1], bounds=DEFAULT_IMAGE_BOUNDS)
    try:
        return _figure_rgb(figure)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def render_cubset_grayscale_image(
    cubset: CubSet,
    camera_pose: tuple[float | None, float | None] = (25.0, -60.0),
) -> torch.Tensor:
    """Render a cubset with :func:`render_cubset_bw` and return a grayscale mask."""
    figure = render_cubset_bw(cubset, elev=camera_pose[0], azim=camera_pose[1], bounds=DEFAULT_IMAGE_BOUNDS)
    try:
        rgb = _figure_rgb(figure)
        return (rgb.mean(dim=-1) < 0.5).to(dtype=torch.float32)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def render_cubset(
    cubset: CubSet,
    *,
    ax: Any | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    elev: float | None = 25.0,
    azim: float | None = -60.0,
    bounds: tuple[float, float, float, float, float, float] | None = None,
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
        # PyTorch3D uses +Y as up and the same spherical-angle convention.
        ax.view_init(elev=elev, azim=azim, vertical_axis=MATPLOTLIB_VERTICAL_AXIS)

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

    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_zlim(bounds[4], bounds[5])
        ax.set_box_aspect((1, 1, 1))
    elif all_vertices:
        vertex_bounds = np.concatenate(all_vertices)
        minimum = vertex_bounds.min(axis=0)
        maximum = vertex_bounds.max(axis=0)
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
    bounds: tuple[float, float, float, float, float, float] | None = None,
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
        ax.view_init(elev=elev, azim=azim, vertical_axis=MATPLOTLIB_VERTICAL_AXIS)
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

    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_zlim(bounds[4], bounds[5])
        ax.set_box_aspect((1, 1, 1))
    elif all_vertices:
        vertex_bounds = np.concatenate(all_vertices)
        minimum = vertex_bounds.min(axis=0)
        maximum = vertex_bounds.max(axis=0)
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
