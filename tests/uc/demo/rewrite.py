"""Show the cuboid grammar rewrites starting from its initial cube.

Run with ``uv run python -m tests.uc.demo.rewrite``. Add ``--save PATH``
to write the gallery to a PNG instead of only opening it interactively.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from .. import (
    AddCub,
    AddHoleXY,
    AddHoleXZ,
    AddHoleYZ,
    CubGrammar,
    RemoveRect,
    SplitX,
    SplitY,
    SplitZ,
    render_cubset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, help="write the rewrite gallery to a PNG")
    args = parser.parse_args()

    grammar = CubGrammar()
    initial = grammar.initial()
    examples = [
        ("initial", initial),
        ("SplitX(i=0, t=0.5)", grammar.apply(initial, SplitX(i=0, t=0.5))),
        ("SplitY(i=0, t=0.5)", grammar.apply(initial, SplitY(i=0, t=0.5))),
        ("SplitZ(i=0, t=0.5)", grammar.apply(initial, SplitZ(i=0, t=0.5))),
        ("AddCub(x=1.2, y=0, z=0, s=0.3)", grammar.apply(initial, AddCub(x=1.2, y=0.0, z=0.0, s=0.3))),
        ("AddHoleXY(i=0, x=0, y=0, s=0.2)", grammar.apply(initial, AddHoleXY(i=0, x=0.0, y=0.0, s=0.2))),
        ("AddHoleXZ(i=0, x=0, z=0, s=0.2)", grammar.apply(initial, AddHoleXZ(i=0, x=0.0, z=0.0, s=0.2))),
        ("AddHoleYZ(i=0, y=0, z=0, s=0.2)", grammar.apply(initial, AddHoleYZ(i=0, y=0.0, z=0.0, s=0.2))),
        ("RemoveRect(i=0)", grammar.apply(initial, RemoveRect(i=0))),
    ]

    figure = plt.figure(figsize=(15, 10))
    for index, (title, cubset) in enumerate(examples, start=1):
        axis = figure.add_subplot(3, 3, index, projection="3d")
        render_cubset(cubset, ax=axis, show=False)
        axis.set_title(f"{title}\n{len(cubset.program)} cuboid(s)", fontsize=9)

    figure.tight_layout()
    if args.save:
        figure.savefig(args.save, dpi=160, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()