from __future__ import annotations

import torch

from .uc_object import denormalize_cuboid_params


def test_denormalize_cuboid_params_uses_scene_scale_and_rotation_scale() -> None:
    params = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.5, 0.25, 1.0, -0.5, 0.0, 0.2, 0.4, 0.6]]
    )

    world = denormalize_cuboid_params(params)

    assert torch.allclose(world[0, :3], torch.tensor([0.0, 1.25, 0.0]))
    assert torch.allclose(world[0, 3:6], torch.tensor([2.0, 1.0, 0.5]))
    assert torch.allclose(world[0, 6:9], torch.tensor([torch.pi, -torch.pi / 2, 0.0]))
    assert torch.equal(world[0, 9:12], params[0, 9:12])


def test_denormalize_cuboid_params_preserves_gradients() -> None:
    params = torch.zeros((1, 12), requires_grad=True)

    denormalize_cuboid_params(params).sum().backward()

    assert torch.allclose(params.grad[0, :3], torch.full((3,), 2.0))
    assert torch.allclose(params.grad[0, 3:6], torch.full((3,), 2.0))
    assert torch.allclose(params.grad[0, 6:9], torch.full((3,), torch.pi))
