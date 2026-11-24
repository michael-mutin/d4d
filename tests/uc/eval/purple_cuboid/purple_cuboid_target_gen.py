"""Generate and save the rendered Purple Cuboid target views."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from .purple_cuboid_target import make_target


def main() -> None:
    output_dir = Path(__file__).with_name("targets")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = make_target()
    for index, (rgb, grayscale) in enumerate(zip(target.rgb_images, target.grayscale_images)):
        iio.imwrite(output_dir / f"view{index:02d}_rgb.png", _uint8(rgb.numpy()))
        iio.imwrite(output_dir / f"view{index:02d}_grayscale.png", _uint8(grayscale.numpy()))
    print(f"wrote {len(target.rgb_images)} views to {output_dir}")


def _uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)


if __name__ == "__main__":
    main()

