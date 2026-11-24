"""Run SRD optimization on the Purple Cuboid target.

Run from the d4d repository with:
``uv run python -m tests.uc.eval.purple_cuboid.purple_cuboid``
"""

from __future__ import annotations

from d4d import OptimizeArgs, optimize

from ...uc_grammar import CubGrammar
from ..util import UCStep
from .purple_cuboid_target import make_target


def main() -> None:
    target = make_target()
    grammar = CubGrammar(
        rgb_images=target.rgb_images,
        grayscale_images=target.grayscale_images,
        camera_poses=target.camera_poses,
    )
    args = OptimizeArgs(
        n_steps=600,
        lr=0.01,
        propose_every=1000,
        w_simplicity=1e-3,
        visualize_every=50,
        seed=0,
    )
    optimize(grammar, args, [UCStep("purple_cuboid", target=target.cuboid)])


if __name__ == "__main__":
    main()
