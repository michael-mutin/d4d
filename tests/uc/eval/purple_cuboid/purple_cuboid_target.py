"""Target object and differentiably rendered views for the purple cuboid."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ... import CubSet, render_cubset_differentiable_grayscale_image, render_cubset_differentiable_rgb_image

IMAGE_SIZE = 64
CAMERA_POSES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (30.0, 45.0),
    (-30.0, 45.0),
    (60.0, 135.0),
    (-60.0, 135.0),
    (20.0, 225.0),
    (45.0, 315.0),
    (-20.0, 315.0),
)


@dataclass(frozen=True)
class PurpleCuboidTarget:
    """A UC target and its RGB/mask views."""

    cuboid: CubSet
    rgb_images: tuple[torch.Tensor, ...]
    grayscale_images: tuple[torch.Tensor, ...]
    camera_poses: tuple[tuple[float, float], ...]


def purple_cuboid() -> CubSet:
    """Return a normalized 3:2:1 purple cuboid rotated by 10, 20, and 40 degrees."""
    return CubSet(
        program="A",
        params=torch.tensor(
            [[
                0.0, 0.0, 0.0,
                0.75, 0.5, 0.25,
                10.0 / 180.0, 20.0 / 180.0, 40.0 / 180.0,
                1.0, 0.0, 1.0,
            ]],
            dtype=torch.float32,
        ),
    )


def make_target(image_size: int = IMAGE_SIZE) -> PurpleCuboidTarget:
    """Render eight RGB and grayscale target images from spherical camera poses."""
    cuboid = purple_cuboid()
    with torch.no_grad():
        rgb_images = tuple(
            render_cubset_differentiable_rgb_image(cuboid, pose, image_size)
            for pose in CAMERA_POSES
        )
        grayscale_images = tuple(
            (render_cubset_differentiable_grayscale_image(cuboid, pose, image_size) > 0.5).to(torch.float32)
            for pose in CAMERA_POSES
        )
    return PurpleCuboidTarget(cuboid, rgb_images, grayscale_images, CAMERA_POSES)
