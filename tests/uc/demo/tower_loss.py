"""Compare colored cube towers with the image loss.

Run with ``uv run python -m tests.uc.demo.tower_loss``. Add ``--save PATH``
to write the comparison figure instead of opening it interactively.
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import torch

from d4d import StepContext

from .. import (
    CubGrammar,
    CubSet,
    render_cubset_grayscale_image,
    render_cubset_rgb_image,
    render_cubset_differentiable_grayscale_image,
    render_cubset_differentiable_rgb_image,
)


def tower(colors: list[tuple[float, float, float]]) -> CubSet:
    """Build a vertical tower from bottom to top."""
    rows: list[list[float]] = []
    for height, color in enumerate(colors):
        rows.append([0.0, height * 1.05, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, *color])
    return CubSet("A" * len(rows), torch.tensor(rows, dtype=torch.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, help="write the comparison figure to a PNG")
    args = parser.parse_args()

    red, blue, yellow = (0.9, 0.05, 0.05), (0.05, 0.2, 0.9), (0.95, 0.8, 0.05)
    ground_truth = tower([red, blue, yellow])
    samples = [
        ("Ground truth\n(uc_rendering)", ground_truth),
        ("Ground truth\n(differentiable)", ground_truth),
        ("Wrong colors", tower([(0.0, 0.0, 0.0)] * 3)),
        ("Missing cube", tower([red, blue])),
    ]

    rgb_target = render_cubset_rgb_image(ground_truth)
    grayscale_target = render_cubset_grayscale_image(ground_truth)
    grammar = CubGrammar(
        rgb_images=[rgb_target],
        grayscale_images=[grayscale_target],
        camera_poses=[(25.0, -60.0)],
    )
    batch = grammar.collate([obj for _, obj in samples])
    ctx = StepContext(step=0, total_steps=1, phase="step")
    losses, metrics = grammar.loss(batch, ctx=ctx, state=None)
    simplicity = grammar.simplicity(batch, ctx)

    figure, axes = plt.subplots(2, len(samples), figsize=(10, 5))
    for index, (name, sample) in enumerate(samples):
        if index == 0:
            rgb_image = render_cubset_rgb_image(sample)
            grayscale_image = render_cubset_grayscale_image(sample)
        else:
            rgb_image = render_cubset_differentiable_rgb_image(sample)
            grayscale_image = render_cubset_differentiable_grayscale_image(sample)
        axes[0, index].imshow(rgb_image)
        axes[0, index].set_title(name)
        axes[1, index].imshow(grayscale_image, cmap="gray", vmin=0, vmax=1)
        axes[1, index].set_xlabel(
            f"Loss: {losses[index].item():.4f}\n"
            f"MSE: {metrics['loss_mse'][index]:.4f} | BCE: {metrics['loss_bce'][index]:.4f}\n"
            f"Simplicity: {simplicity[index]:.0f} cuboids"
        )
        for axis in axes[:, index]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_ylabel("RGB\n(first: uc_rendering)")
    axes[1, 0].set_ylabel("Grayscale\n(first: uc_rendering)")
    figure.suptitle(
        "Targets used for the loss: RGB and grayscale rendered with the non-differentiable renderer"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    if args.save:
        figure.savefig(args.save, dpi=160, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()