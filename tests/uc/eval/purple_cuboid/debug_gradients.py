"""Measure purple-cuboid gradients for several differentiable rasterizer settings.

Run from the d4d repository, for example:

``uv run python -m tests.uc.eval.purple_cuboid.debug_gradients --steps 100``

The script writes one PNG per measured quantity. Each PNG contains heatmaps for
all normalized parameters, using one shared color scale for direct comparison.
The x-axis is the optimization step and the y-axis is the blur radius.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ...uc_grammar import CubGrammar
from .purple_cuboid_target import make_target

PARAMETER_NAMES = (
    "x", "y", "z", "half_x", "half_y", "half_z",
    "rotation_x", "rotation_y", "rotation_z", "red", "green", "blue",
)
DEFAULT_BLUR_RADII = (1e-4, 1e-3, 1e-2, 5e-2)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "eval_out" / "purple_cuboid_gradients"
)


class DebugPurpleCuboidGrammar(CubGrammar):
    """Cuboid grammar with no rewrite proposals for gradient measurements."""

    def propose(self, obj, budget):
        return []


def parse_blur_radii(value: str) -> tuple[float, ...]:
    radii = tuple(float(item) for item in value.split(",") if item.strip())
    if not radii or any(radius < 0 for radius in radii):
        raise argparse.ArgumentTypeError("blur radii must be a non-empty list of non-negative numbers")
    return radii


def measure(
    *,
    target,
    blur_radii: tuple[float, ...],
    steps: int,
    lr: float,
    optimizer_name: str,
) -> dict[str, np.ndarray]:
    values = np.zeros((4, len(blur_radii), steps + 1, 12), dtype=np.float64)
    values[3].fill(np.nan)

    total_iterations = len(blur_radii) * (steps + 1)
    with tqdm(
        total=total_iterations,
        desc="Measuring gradients",
        unit="step",
    ) as progress:
        for blur_index, blur_radius in enumerate(blur_radii):
            grammar = DebugPurpleCuboidGrammar(
                rgb_images=target.rgb_images,
                grayscale_images=target.grayscale_images,
                camera_poses=target.camera_poses,
                blur_radius=blur_radius,
            )
            batch = grammar.collate([grammar.initial()]).requires_grad_()
            parameters = batch.parameters()[0]
            optimizer_class = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}[optimizer_name]
            optimizer = optimizer_class(batch.parameters(), lr=lr)

            for step in range(steps + 1):
                optimizer.zero_grad()
                mse, bce = grammar.loss_differentiable_components(batch)
                mse_grad = torch.autograd.grad(mse.sum(), parameters, retain_graph=True)[0]
                bce_grad = torch.autograd.grad(bce.sum(), parameters, retain_graph=True)[0]
                total = mse + bce
                total_grad = torch.autograd.grad(total.sum(), parameters, retain_graph=True)[0]

                values[0, blur_index, step] = mse_grad.detach().cpu().numpy().reshape(-1)
                values[1, blur_index, step] = bce_grad.detach().cpu().numpy().reshape(-1)
                values[2, blur_index, step] = total_grad.detach().cpu().numpy().reshape(-1)

                if step < steps:
                    before = parameters.detach().clone()
                    total.sum().backward()
                    optimizer.step()
                    values[3, blur_index, step] = (
                        parameters.detach() - before
                    ).cpu().numpy().reshape(-1)

                progress.update(1)
                progress.set_postfix(blur=f"{blur_radius:g}", refresh=False)

    return {
        "mse_gradient": values[0],
        "bce_gradient": values[1],
        "total_gradient": values[2],
        "update": values[3],
    }


def save_plots(
    measurements: dict[str, np.ndarray],
    blur_radii: tuple[float, ...],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "mse_gradient": "RGB-MSE gradient",
        "bce_gradient": "mask-BCE gradient",
        "total_gradient": "total gradient",
        "update": "parameter update",
    }

    for key, label in labels.items():
        data = measurements[key]
        figure, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
        for parameter_index, (axis, parameter_name) in enumerate(
            zip(axes.flat, PARAMETER_NAMES)
        ):
            parameter_data = data[:, :, parameter_index]
            finite = np.abs(parameter_data[np.isfinite(parameter_data)])
            scale = max(float(np.max(finite)) if finite.size else 0.0, 1e-12)
            image = axis.imshow(
                parameter_data,
                aspect="auto",
                origin="upper",
                cmap="coolwarm",
                vmin=-scale,
                vmax=scale,
                interpolation="nearest",
            )
            axis.set_title(parameter_name)
            axis.set_xlabel("optimization step")
            axis.set_ylabel("blur radius")
            axis.set_yticks(range(len(blur_radii)))
            axis.set_yticklabels([f"{radius:g}" for radius in blur_radii])
            figure.colorbar(image, ax=axis, label="value", fraction=0.046, pad=0.04)

        figure.suptitle(f"Purple cuboid: {label} (individual color scales)")
        figure.savefig(output_dir / f"{key}.png", dpi=160)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("Adam", "SGD"), default="Adam")
    parser.add_argument("--blur-radii", type=parse_blur_radii, default=DEFAULT_BLUR_RADII)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    output_dir = args.output_dir.resolve()
    target = make_target()
    measurements = measure(
        target=target,
        blur_radii=args.blur_radii,
        steps=args.steps,
        lr=args.lr,
        optimizer_name=args.optimizer,
    )
    save_plots(measurements, args.blur_radii, output_dir)
    print(f"Saved {len(measurements)} diagnostic grids to {output_dir}")


if __name__ == "__main__":
    main()
