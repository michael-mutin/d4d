"""Render a solved Rubik's cube.

Run with ``uv run python -m tests.uc.demo.rubics`` and drag in the window to
move the camera. Add ``--save`` to also write a PNG.
"""

from __future__ import annotations

import argparse

import torch

from .. import CubSet, render_cubset


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
    render_cubset(solved_cube(), save_path=args.save, show=True)


if __name__ == "__main__":
    main()
