"""Run SRD optimization on the Purple Cuboid target.

Run from the d4d repository with:
``uv run python -m tests.uc.eval.purple_cuboid.purple_cuboid``
"""

from __future__ import annotations

import numpy as np
import torch

from d4d import OptimizeArgs, optimize

from ... import render_cubset_differentiable_rgb_image
from ...uc_grammar import CubGrammar
from ..util import UCStep
from .purple_cuboid_target import make_target


class PurpleCuboidGrammar(CubGrammar):
    """UC grammar configured with the currently implemented random proposals."""

    def propose(self, obj, budget):
        return self.propose_random(obj, budget)

    def visualize(self, batch, ctx, state):
        if not self.rgb_images or not self.camera_poses or len(batch) == 0:
            return None

        def to_uint8(image: torch.Tensor) -> np.ndarray:
            return (image.clamp(0.0, 1.0) * 255).round().to(dtype=torch.uint8).numpy()

        frames = []
        for target_image, camera_pose in zip(self.rgb_images, self.camera_poses):
            target = self._rgb_tensor(target_image, batch.get(0).params).detach().cpu()
            current = render_cubset_differentiable_rgb_image(
                batch.get(0),
                camera_pose,
                self._rgb_resolution(target),
            ).detach().cpu()
            frames.append(np.concatenate((to_uint8(target), to_uint8(current)), axis=1))

        rows = [np.concatenate(frames[index:index + 2], axis=1) for index in range(0, len(frames), 2)]
        return np.concatenate(rows, axis=0)


def main() -> None:
    target = make_target()
    grammar = PurpleCuboidGrammar(
        rgb_images=target.rgb_images,
        grayscale_images=target.grayscale_images,
        camera_poses=target.camera_poses,
    )
    args = OptimizeArgs(
        n_steps=600,
        lr=0.1,
        propose_every=1000,
        w_simplicity=1e-3,
        visualize_every=50,
        seed=0,
    )
    optimize(grammar, args, [UCStep("purple_cuboid", target=target.cuboid)])


if __name__ == "__main__":
    main()
