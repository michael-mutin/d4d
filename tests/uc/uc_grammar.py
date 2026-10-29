"""The parametric union-cuboid shape grammar."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from d4d import ExtraMetrics, Grammar, ListCollection, ListSpec, StepContext

from .uc_object import CubSet
from .uc_differentiable_rendering import (
    render_cubset_differentiable_grayscale_image,
    render_cubset_differentiable_rgb_image,
)
from .uc_rendering import render_cubset_rgb_image
from .uc_rewrite import AddCub, AddHoleXY, AddHoleXZ, AddHoleYZ, Merge, RemoveRect, Rewrite, SplitX, SplitY, SplitZ


SIMPLICITY_SCALE = 0.01


class CubGrammar(Grammar[CubSet, ListCollection[CubSet], Rewrite, None]):
    """Grammar for parametric union-cuboid objects and image-view targets."""

    list_spec = ListSpec(
        params_of=lambda o: [o.params],
        with_params=lambda o, ts: replace(o, params=ts[0]),
        names=("params",),
    )

    def __init__(
        self,
        rgb_images: Sequence[torch.Tensor] | None = None,
        grayscale_images: Sequence[torch.Tensor] | None = None,
        camera_poses: Sequence[Any] | None = None,
        render_rgb: Callable[[CubSet, Any], torch.Tensor] | None = None,
        render_grayscale: Callable[[CubSet, Any], torch.Tensor] | None = None,
    ) -> None:
        """Store image targets and optional renderers for the non-differentiable loss."""
        self.rgb_images: tuple[torch.Tensor, ...]
        self.grayscale_images: tuple[torch.Tensor, ...]
        self.camera_poses: tuple[Any, ...]
        if rgb_images is None and grayscale_images is None and camera_poses is None:
            self.rgb_images = ()
            self.grayscale_images = ()
            self.camera_poses = ()
        else:
            if rgb_images is None or grayscale_images is None or camera_poses is None:
                raise ValueError("RGB images, grayscale images, and camera poses must be provided together")
            if len(rgb_images) == 0 or len(grayscale_images) == 0 or len(camera_poses) == 0:
                raise ValueError("at least one RGB image, grayscale image, and camera pose is required")
            if not len(rgb_images) == len(grayscale_images) == len(camera_poses):
                raise ValueError("RGB images, grayscale images, and camera poses must have equal lengths")

            rgb_resolution = self._rgb_resolution(rgb_images[0])
            for image in rgb_images:
                if self._rgb_resolution(image) != rgb_resolution:
                    raise ValueError("all RGB images must have the same resolution")

            for image in grayscale_images:
                if self._grayscale_resolution(image) != rgb_resolution:
                    raise ValueError("RGB and grayscale images must have the same resolution")
                if not torch.all((image == 0) | (image == 1) | (image == 255)):
                    raise ValueError("grayscale images must contain only black/white values (0, 1, or 255)")

            self.rgb_images = tuple(rgb_images)
            self.grayscale_images = tuple(grayscale_images)
            self.camera_poses = tuple(camera_poses)
        self.render_rgb = render_rgb
        self.render_grayscale = render_grayscale

    @staticmethod
    def _rgb_resolution(image: torch.Tensor) -> tuple[int, int]:
        if image.ndim != 3:
            raise ValueError("RGB images must have shape (H, W, 3) or (3, H, W)")
        if image.shape[-1] == 3:
            return int(image.shape[0]), int(image.shape[1])
        if image.shape[0] == 3:
            return int(image.shape[1]), int(image.shape[2])
        raise ValueError("RGB images must have shape (H, W, 3) or (3, H, W)")

    @staticmethod
    def _grayscale_resolution(image: torch.Tensor) -> tuple[int, int]:
        if image.ndim == 2:
            return int(image.shape[0]), int(image.shape[1])
        if image.ndim == 3 and image.shape[-1] == 1:
            return int(image.shape[0]), int(image.shape[1])
        if image.ndim == 3 and image.shape[0] == 1:
            return int(image.shape[1]), int(image.shape[2])
        raise ValueError("grayscale images must have shape (H, W), (H, W, 1), or (1, H, W)")

    def initial(self) -> CubSet:
        return CubSet(
            program="A",
            params=torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]],
                dtype=torch.float32,
            ),
        )

    def propose(self, obj: CubSet, budget: int) -> list[Rewrite]:
        return self.propose_smart(obj, budget)

    def propose_random(self, obj: CubSet, budget: int) -> list[Rewrite]:
        """Create a small pool of random rewrite candidates."""
        values = obj.params.detach().tolist()
        rewrites: list[Rewrite] = []

        for i, row in enumerate(values):
            x, y, z, sx, sy, sz = row[:6]
            rewrites.extend(
                (
                    SplitX(i, random.uniform(0.1, 0.9)),
                    SplitY(i, random.uniform(0.1, 0.9)),
                    SplitZ(i, random.uniform(0.1, 0.9)),
                    AddHoleXY(i, x, y, random.uniform(0.1, 1.0) * min(sx, sy)),
                    AddHoleXZ(i, x, z, random.uniform(0.1, 1.0) * min(sx, sz)),
                    AddHoleYZ(i, y, z, random.uniform(0.1, 1.0) * min(sy, sz)),
                )
            )

        if values:
            rewrites.append(
                AddCub(
                    random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0),
                    random.uniform(0.1, 1.0),
                )
            )
        if len(values) > 1:
            rewrites.extend(RemoveRect(i) for i in range(len(values)))
            rewrites.extend(Merge(i, j) for i in range(len(values)) for j in range(i + 1, len(values)))

        if budget > 0 and len(rewrites) > budget:
            return random.sample(rewrites, budget)
        return rewrites

    def propose_smart(self, obj: CubSet, budget: int) -> list[Rewrite]:
        raise NotImplementedError

    def visualize(
        self, batch: ListCollection[CubSet], ctx: StepContext, state: None
    ) -> np.ndarray | None:
        """Return the RGB target and current shape next to each other."""
        if not self.rgb_images or len(batch) == 0:
            return None

        target = self._rgb_tensor(self.rgb_images[0], batch.get(0).params).detach().cpu()
        current = render_cubset_rgb_image(batch.get(0), self.camera_poses[0]).detach().cpu()
        if current.shape != target.shape:
            raise ValueError(
                f"visualization target and rendered shape must have the same resolution, "
                f"got {tuple(target.shape)} and {tuple(current.shape)}"
            )

        def to_uint8(image: torch.Tensor) -> np.ndarray:
            if torch.any(image > 1):
                image = image / 255.0
            return (image.clamp(0.0, 1.0) * 255.0).round().to(dtype=torch.uint8).numpy()

        return np.concatenate((to_uint8(target), to_uint8(current)), axis=1)

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
        def rightmost_index(rewrite: Rewrite) -> int:
            if isinstance(rewrite, AddCub):
                return -1
            if isinstance(rewrite, Merge):
                return max(rewrite.i, rewrite.j)
            return rewrite.i

        out = base
        for rewrite in sorted(rewrites, key=rightmost_index, reverse=True):
            out = self.apply(out, rewrite)
        return out

    def loss(self, batch: ListCollection[CubSet], ctx: StepContext, state: None) -> tuple[torch.Tensor, ExtraMetrics]:
        return self.loss_differentiable(batch, ctx, state)

    def loss_non_differentiable(
        self, batch: ListCollection[CubSet], ctx: StepContext, state: None
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        """Evaluate the image loss using the currently available renderers."""
        if self.rgb_images and self.render_rgb is None:
            raise RuntimeError("render_rgb is required when RGB target images are configured")
        if self.grayscale_images and self.render_grayscale is None:
            raise RuntimeError("render_grayscale is required when grayscale target images are configured")

        losses: list[torch.Tensor] = []
        mse_values: list[float] = []
        bce_values: list[float] = []
        for obj in batch:
            reference = obj.params
            mse = reference.new_zeros(())
            bce = reference.new_zeros(())

            if self.rgb_images:
                assert self.render_rgb is not None
                mse_terms = []
                for target, camera_pose in zip(self.rgb_images, self.camera_poses):
                    prediction = self._rgb_tensor(self.render_rgb(obj, camera_pose), reference)
                    expected = self._rgb_tensor(target, reference)
                    mse_terms.append(torch.mean((prediction - expected) ** 2))
                mse = torch.stack(mse_terms).mean()

            if self.grayscale_images:
                assert self.render_grayscale is not None
                bce_terms = []
                for target, camera_pose in zip(self.grayscale_images, self.camera_poses):
                    prediction = self._grayscale_tensor(self.render_grayscale(obj, camera_pose), reference)
                    expected = self._grayscale_tensor(target, reference)
                    epsilon = torch.finfo(prediction.dtype).eps
                    prediction = prediction.clamp(epsilon, 1.0 - epsilon)
                    bce_terms.append(torch.nn.functional.binary_cross_entropy(prediction, expected))
                bce = torch.stack(bce_terms).mean()

            losses.append(mse + bce)
            mse_values.append(float(mse.detach()))
            bce_values.append(float(bce.detach()))

        return torch.stack(losses), {"loss_mse": mse_values, "loss_bce": bce_values}

    def loss_differentiable(
        self, batch: ListCollection[CubSet], ctx: StepContext, state: None
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        losses: list[torch.Tensor] = []
        mse_values: list[float] = []
        bce_values: list[float] = []
        for index in range(len(batch)):
            obj = batch.get(index, detach=False)
            reference = obj.params
            mse = reference.new_zeros(())
            bce = reference.new_zeros(())

            if self.rgb_images:
                mse_terms = []
                for target, camera_pose in zip(self.rgb_images, self.camera_poses):
                    prediction = render_cubset_differentiable_rgb_image(
                        obj, camera_pose, self._rgb_resolution(target)
                    )
                    prediction = self._rgb_tensor(prediction, reference)
                    expected = self._rgb_tensor(target, reference)
                    mse_terms.append(torch.mean((prediction - expected) ** 2))
                mse = torch.stack(mse_terms).mean()

            if self.grayscale_images:
                bce_terms = []
                for target, camera_pose in zip(self.grayscale_images, self.camera_poses):
                    prediction = render_cubset_differentiable_grayscale_image(
                        obj, camera_pose, self._grayscale_resolution(target)
                    )
                    prediction = self._grayscale_tensor(prediction, reference)
                    expected = self._grayscale_tensor(target, reference)
                    epsilon = torch.finfo(prediction.dtype).eps
                    bce_terms.append(torch.nn.functional.binary_cross_entropy(prediction.clamp(epsilon, 1.0 - epsilon), expected))
                bce = torch.stack(bce_terms).mean()

            losses.append(mse + bce)
            mse_values.append(float(mse.detach()))
            bce_values.append(float(bce.detach()))

        return torch.stack(losses), {"loss_mse": mse_values, "loss_bce": bce_values}

    @staticmethod
    def _rgb_tensor(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError("RGB renderer output must have shape (H, W, 3) or (3, H, W)")
        if image.shape[-1] == 3:
            pass
        elif image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        else:
            raise ValueError("RGB renderer output must have shape (H, W, 3) or (3, H, W)")
        return image.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _grayscale_tensor(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        elif image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        elif image.ndim != 2:
            raise ValueError("grayscale renderer output must have shape (H, W), (H, W, 1), or (1, H, W)")
        image = image.to(device=reference.device, dtype=reference.dtype)
        if torch.any(image > 1):
            image = image / 255.0
        return image

    def simplicity(self, batch: ListCollection[CubSet], ctx: StepContext) -> list[float]:
        return [SIMPLICITY_SCALE * float(len(obj.params)) for obj in batch]
