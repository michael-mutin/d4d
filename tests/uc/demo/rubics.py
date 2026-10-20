"""Render a solved Rubik's cube.

Run with ``uv run python -m tests.uc.demo.rubics`` and drag in the window to
move the camera. Add ``--save`` to also write a PNG.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from .. import CubSet, render_cubset, render_cubset_bw


def solved_cube() -> CubSet:
    """Create a solved 3x3x3 cube from cuboids and thin colored stickers."""
    rows: list[list[float]] = []
    coordinates = (-1.0, 0.0, 1.0)
    sticker_colors = {
        "x-": (1.0, 0.45, 0.05),
        "x+": (0.85, 0.05, 0.05),
        "y-": (1.0, 0.85, 0.05),
        "y+": (1.0, 1.0, 1.0),
        "z-": (0.05, 0.2, 0.85),
        "z+": (0.05, 0.65, 0.2),
    }

    for x in coordinates:
        for y in coordinates:
            for z in coordinates:
                rows.append([x, y, z, 0.49, 0.49, 0.49, 0, 0, 0, 0.035, 0.035, 0.035])
                if x == coordinates[0]:
                    rows.append([x - 0.495, y, z, 0.04, 0.46, 0.46, 0, 0, 0, *sticker_colors["x-"]])
                if x == coordinates[-1]:
                    rows.append([x + 0.495, y, z, 0.04, 0.46, 0.46, 0, 0, 0, *sticker_colors["x+"]])
                if y == coordinates[0]:
                    rows.append([x, y - 0.495, z, 0.46, 0.04, 0.46, 0, 0, 0, *sticker_colors["y-"]])
                if y == coordinates[-1]:
                    rows.append([x, y + 0.495, z, 0.46, 0.04, 0.46, 0, 0, 0, *sticker_colors["y+"]])
                if z == coordinates[0]:
                    rows.append([x, y, z - 0.495, 0.46, 0.46, 0.04, 0, 0, 0, *sticker_colors["z-"]])
                if z == coordinates[-1]:
                    rows.append([x, y, z + 0.495, 0.46, 0.46, 0.04, 0, 0, 0, *sticker_colors["z+"]])

    return CubSet("A" * len(rows), torch.tensor(rows, dtype=torch.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, help="also save the rendered cube as a PNG")
    args = parser.parse_args()
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"})
    cube = solved_cube()
    render_cubset(cube, ax=axes[0], show=False)
    render_cubset_bw(cube, ax=axes[1], show=False)
    axes[0].set_title("Farbe")
    axes[1].set_title("Schwarz-Weiß")
    if args.save:
        figure.savefig(args.save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
