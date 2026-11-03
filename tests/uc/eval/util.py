"""Reusable callbacks and result handling for UC experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from d4d import Callback, RunEnd, StepEnd, VisualizeEvent
from ..uc_differentiable_rendering import write_cubset_obj


class UCStep(Callback):
    """Print optimization progress and persist images and loss history."""

    def __init__(self, experiment: str, output_root: str | Path | None = None) -> None:
        root = Path(output_root) if output_root is not None else Path(__file__).parents[1] / "eval_out"
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        self.output_dir = root / experiment / timestamp
        self.losses: list[float] = []
        self.losses_cont: list[float] = []
        self.losses_simplicity: list[float] = []
        self.losses_ma: list[float] = []
        self.extra: dict[str, list[float]] = {}

    def on_step_end(self, ev: StepEnd) -> None:
        self.losses.append(ev.loss)
        self.losses_cont.append(ev.loss_cont)
        self.losses_simplicity.append(ev.loss_simplicity)
        self.losses_ma.append(ev.loss_ma)
        for name, value in ev.extra.items():
            self.extra.setdefault(name, []).append(float(value))
        if ev.rewrite or ev.step == 0:
            obj = ev.get_object()
            print(
                f"step {ev.step:4d}  cuboids {len(obj.params):4d}"
                f"  loss {ev.loss_cont:.6f}  loss_ma {ev.loss_ma:.6f}"
            )

    def on_visualize(self, ev: VisualizeEvent) -> None:
        import imageio.v3 as iio

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image = np.asarray(ev.image)
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)
        iio.imwrite(self.output_dir / f"step{ev.step:06d}.png", image)

    def on_run_end(self, ev: RunEnd) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_plot()
        if ev.result is None:
            return
        best_result = f"best loss: {ev.result.best_loss:.9f}\nbest step: {ev.result.best_step}\n"
        best_result += f"best object:\n{ev.result.best}\n"
        (self.output_dir / "best_result.txt").write_text(best_result, encoding="utf-8")
        self._write_shape(ev.result.best, "best_shape")
        self._write_shape(ev.result.final, "final_shape")
        print(f"best loss {ev.result.best_loss:.9f} (step {ev.result.best_step})")
        print(f"best_result:\n{ev.result.best}")

    def _write_shape(self, obj: Any, stem: str) -> None:
        import pickle

        with (self.output_dir / f"{stem}.pkl").open("wb") as file:
            pickle.dump(obj, file)
        write_cubset_obj(obj, self.output_dir / f"{stem}.obj")

    def _write_plot(self) -> None:
        import matplotlib.pyplot as plt

        if not self.losses:
            return
        figure, axis = plt.subplots(figsize=(9, 5))
        steps = np.arange(len(self.losses))
        axis.plot(steps, self.losses, label="loss")
        axis.plot(steps, self.losses_cont, label="continuous loss")
        axis.plot(steps, self.losses_ma, label="moving average")
        if any(self.losses_simplicity):
            axis.plot(steps, self.losses_simplicity, label="simplicity")
        for name, values in self.extra.items():
            if len(values) == len(steps):
                axis.plot(steps, values, label=name, linestyle="--")
        axis.set(xlabel="step", ylabel="value", title="Purple cuboid optimization")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "losses.png", dpi=160)
        plt.close(figure)


def best_loss(result: Any) -> float:
    """Print and return the best loss of an optimization result."""
    value = float(result.best_loss)
    print(f"best loss {value:.9f} (step {result.best_step})")
    return value
