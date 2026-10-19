"""The parametric union-cuboid shape grammar."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from d4d import ExtraMetrics, Grammar, ListCollection, ListSpec, StepContext

from .uc_object import CubSet
from .uc_rewrite import AddCub, AddHoleXY, AddHoleXZ, AddHoleYZ, Merge, RemoveRect, Rewrite, SplitX, SplitY, SplitZ


class CubGrammar(Grammar[CubSet, ListCollection[CubSet], Rewrite, None]):
    """Placeholder grammar for parametric union-cuboid objects."""

    list_spec = ListSpec(
        params_of=lambda o: [o.params],
        with_params=lambda o, ts: replace(o, params=ts[0]),
        names=("params",),
    )

    def initial(self) -> CubSet:
        return CubSet(
            program="A",
            params=torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]],
                dtype=torch.float32,
            ),
        )

    def propose(self, obj: CubSet, budget: int) -> list[Rewrite]:
        raise NotImplementedError

    def apply(self, obj: CubSet, rewrite: Rewrite) -> CubSet:
        values = obj.params.detach().clone()

        def replace_rows(index: int, rows: list[torch.Tensor]) -> CubSet:
            params = torch.cat([values[:index], *(row.unsqueeze(0) for row in rows), values[index + 1 :]])
            return CubSet(
                program=obj.program[:index] + "A" * len(rows) + obj.program[index + 1 :],
                params=params,
            )

        def split(index: int, axis: int, fraction: float) -> CubSet:
            if not 0.0 < fraction < 1.0:
                raise ValueError("split position must be between 0 and 1")
            row = values[index]
            first = row.clone()
            second = row.clone()
            size = row[3 + axis]
            center = row[axis]
            first[3 + axis] = size * fraction
            second[3 + axis] = size * (1.0 - fraction)
            first[axis] = center - size + first[3 + axis]
            second[axis] = center + size - second[3 + axis]
            return replace_rows(index, [first, second])

        def cut_hole(index: int, axis_a: int, coordinate_a: float, axis_b: int, coordinate_b: float, radius: float) -> CubSet:
            if radius < 0.0:
                raise ValueError("hole radius must be non-negative")
            row = values[index]
            original = {
                axis: (float(row[axis] - row[3 + axis]), float(row[axis] + row[3 + axis]))
                for axis in (axis_a, axis_b)
            }
            hole = {
                axis_a: (max(original[axis_a][0], coordinate_a - radius), min(original[axis_a][1], coordinate_a + radius)),
                axis_b: (max(original[axis_b][0], coordinate_b - radius), min(original[axis_b][1], coordinate_b + radius)),
            }
            if any(lower >= upper for lower, upper in hole.values()):
                return replace_rows(index, [row.clone()])

            pieces: list[torch.Tensor] = []

            def add_piece(range_a: tuple[float, float], range_b: tuple[float, float]) -> None:
                if range_a[0] >= range_a[1] or range_b[0] >= range_b[1]:
                    return
                piece = row.clone()
                for axis, bounds in ((axis_a, range_a), (axis_b, range_b)):
                    piece[axis] = (bounds[0] + bounds[1]) / 2.0
                    piece[3 + axis] = (bounds[1] - bounds[0]) / 2.0
                pieces.append(piece)

            add_piece((original[axis_a][0], hole[axis_a][0]), original[axis_b])
            add_piece((hole[axis_a][1], original[axis_a][1]), original[axis_b])
            add_piece(hole[axis_a], (original[axis_b][0], hole[axis_b][0]))
            add_piece(hole[axis_a], (hole[axis_b][1], original[axis_b][1]))
            return replace_rows(index, pieces)

        if isinstance(rewrite, SplitX):
            return split(rewrite.i, 0, rewrite.t)
        if isinstance(rewrite, SplitY):
            return split(rewrite.i, 1, rewrite.t)
        if isinstance(rewrite, SplitZ):
            return split(rewrite.i, 2, rewrite.t)
        if isinstance(rewrite, AddCub):
            row = values.new_tensor([rewrite.x, rewrite.y, rewrite.z, rewrite.s, rewrite.s, rewrite.s, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5])
            return CubSet(program=obj.program + "A", params=torch.cat([values, row.unsqueeze(0)]))
        if isinstance(rewrite, AddHoleXY):
            return cut_hole(rewrite.i, 0, rewrite.x, 1, rewrite.y, rewrite.s)
        if isinstance(rewrite, AddHoleXZ):
            return cut_hole(rewrite.i, 0, rewrite.x, 2, rewrite.z, rewrite.s)
        if isinstance(rewrite, AddHoleYZ):
            return cut_hole(rewrite.i, 1, rewrite.y, 2, rewrite.z, rewrite.s)
        if isinstance(rewrite, RemoveRect):
            return CubSet(
                program=obj.program[: rewrite.i] + obj.program[rewrite.i + 1 :],
                params=torch.cat([values[: rewrite.i], values[rewrite.i + 1 :]]),
            )
        if isinstance(rewrite, Merge):
            if rewrite.i == rewrite.j:
                raise ValueError("cannot merge a cuboid with itself")
            first, second = sorted((rewrite.i, rewrite.j))
            left, right = values[first], values[second]
            minimum = torch.minimum(left[:3] - left[3:6], right[:3] - right[3:6])
            maximum = torch.maximum(left[:3] + left[3:6], right[:3] + right[3:6])
            merged = left.clone()
            merged[:3] = (minimum + maximum) / 2.0
            merged[3:6] = (maximum - minimum) / 2.0
            params = torch.cat([values[:first], merged.unsqueeze(0), values[first + 1 : second], values[second + 1 :]])
            return CubSet(
                program=obj.program[:first] + "A" + obj.program[first + 1 : second] + obj.program[second + 1 :],
                params=params,
            )
        raise ValueError(f"unknown rewrite {rewrite!r}")

    def conflicts(self, a: Rewrite, b: Rewrite) -> bool:
        def touched(rewrite: Rewrite) -> set[int]:
            if isinstance(rewrite, AddCub):
                return set()
            if isinstance(rewrite, Merge):
                return {rewrite.i, rewrite.j}
            return {rewrite.i}

        return bool(touched(a) & touched(b))

    def apply_all(self, base: CubSet, rewrites: Sequence[Rewrite]) -> CubSet:
        raise NotImplementedError

    def loss(self, batch: ListCollection[CubSet], ctx: StepContext, state: None) -> tuple[torch.Tensor, ExtraMetrics]:
        raise NotImplementedError

    def simplicity(self, batch: ListCollection[CubSet], ctx: StepContext) -> list[float]:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        raise NotImplementedError
