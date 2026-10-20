"""Compare colored cube towers with the image loss.

Run with ``uv run python -m tests.uc.demo.tower_loss``. Add ``--save PATH``
to write the comparison figure instead of opening it interactively.
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import torch

from d4d import StepContext

from .. import CubGrammar, CubSet, render_cubset_grayscale_image, render_cubset_rgb_image


def tower(colors: list[tuple[float, float, float]]) -> CubSet:
    """Build a vertical tower from bottom to top."""
    rows: list[list[float]] = []
    for height, color in enumerate(colors):
        rows.append([0.0, 0.0, height * 1.05, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, *color])
    return CubSet("A" * len(rows), torch.tensor(rows, dtype=torch.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, help="write the comparison figure to a PNG")
    args = parser.parse_args()

    red, blue, yellow = (0.9, 0.05, 0.05), (0.05, 0.2, 0.9), (0.95, 0.8, 0.05)
    ground_truth = tower([red, blue, yellow])
    samples = [
        ("Ground truth", ground_truth),
        ("Wrong colors", tower([(0.0, 0.0, 0.0)] * 3)),
        ("Missing cube", tower([red, blue])),
    ]

    grammar = CubGrammar(
        rgb_images=[render_cubset_rgb_image(ground_truth)],
        grayscale_images=[render_cubset_grayscale_image(ground_truth)],
        camera_poses=[(25.0, -60.0)],
        render_rgb=render_cubset_rgb_image,
        render_grayscale=render_cubset_grayscale_image,
    )
    losses, metrics = grammar.loss(
        grammar.collate([obj for _, obj in samples]),
        ctx=StepContext(step=0, total_steps=1, phase="step"),
        state=None,
    )

    figure, axes = plt.subplots(2, len(samples), figsize=(10, 5))
    for index, (name, _) in enumerate(samples):
        axes[0, index].imshow(render_cubset_rgb_image(samples[index][1]))
        axes[0, index].set_title(name)
        axes[1, index].imshow(render_cubset_grayscale_image(samples[index][1]), cmap="gray", vmin=0, vmax=1)
        axes[1, index].set_xlabel(
            f"Loss: {losses[index].item():.4f}\n"
            f"MSE: {metrics['loss_mse'][index]:.4f} | BCE: {metrics['loss_bce'][index]:.4f}"
        )
        for axis in axes[:, index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    if args.save:
        figure.savefig(args.save, dpi=160, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()