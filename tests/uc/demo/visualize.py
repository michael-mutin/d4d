"""Display the frame produced by :meth:`CubGrammar.visualize`.

Run with ``uv run python -m tests.uc.demo.visualize``. Add ``--save PATH``
to write the frame to a PNG instead of opening it interactively.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from d4d import StepContext

from .. import CubGrammar, CubSet, render_cubset_grayscale_image, render_cubset_rgb_image


def tower() -> CubSet:
    """Build a small colored tower to use as the visualization target."""
    colors = [(0.9, 0.05, 0.05), (0.05, 0.2, 0.9), (0.95, 0.8, 0.05)]
    rows = [
        [0.0, height * 1.05, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, *color]
        for height, color in enumerate(colors)
    ]
    return CubSet("A" * len(rows), torch.tensor(rows, dtype=torch.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, help="write the visualization frame to a PNG")
    args = parser.parse_args()

    target = tower()
    grammar = CubGrammar(
        rgb_images=[render_cubset_rgb_image(target)],
        grayscale_images=[render_cubset_grayscale_image(target)],
        camera_poses=[(25.0, -60.0)],
    )
    current = grammar.initial()
    batch = grammar.collate([current])
    frame = grammar.visualize(
        batch,
        StepContext(step=0, total_steps=1, phase="visualize"),
        state=None,
    )
    if frame is None:
        raise RuntimeError("CubGrammar.visualize returned no image")

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.imshow(frame)
    axis.set_title("Target (left) | Current shape (right)")
    axis.set_axis_off()
    figure.tight_layout()
    if args.save:
        figure.savefig(args.save, dpi=160, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
