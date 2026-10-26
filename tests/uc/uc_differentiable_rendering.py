"""Differentiable PyTorch3D rendering for union-cuboid objects."""

from __future__ import annotations

import torch

from .uc_object import CubSet

DEFAULT_IMAGE_SIZE = (128, 128)
DEFAULT_CAMERA_TARGET = (0.0, 1.25, 0.0)
DEFAULT_CAMERA_DISTANCE = 7.0


def cubset_to_mesh(cubset: CubSet) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create differentiable cuboid vertices, faces, and vertex colors."""
    params = cubset.params
    if params.ndim != 2 or params.shape[1] != 12:
        raise ValueError("CubSet.params must have shape (number_of_cuboids, 12)")
    if len(cubset.program) != params.shape[0]:
        raise ValueError("CubSet.program and CubSet.params must have the same length")

    device, dtype = params.device, params.dtype
    signs = torch.tensor(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        device=device,
        dtype=dtype,
    )
    face_template = torch.tensor(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        device=device,
        dtype=torch.int64,
    )

    centers, half_sizes, angles, colors = params[:, :3], params[:, 3:6], params[:, 6:9], params[:, 9:12]
    alpha, beta, gamma = angles.unbind(dim=1)
    zeros = torch.zeros_like(alpha)
    ones = torch.ones_like(alpha)
    rotate_x = torch.stack(
        (
            ones,
            zeros,
            zeros,
            zeros,
            alpha.cos(),
            -alpha.sin(),
            zeros,
            alpha.sin(),
            alpha.cos(),
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    rotate_y = torch.stack(
        (
            beta.cos(),
            zeros,
            beta.sin(),
            zeros,
            ones,
            zeros,
            -beta.sin(),
            zeros,
            beta.cos(),
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    rotate_z = torch.stack(
        (
            gamma.cos(),
            -gamma.sin(),
            zeros,
            gamma.sin(),
            gamma.cos(),
            zeros,
            zeros,
            zeros,
            ones,
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    rotation = rotate_z @ rotate_y @ rotate_x

    vertices = (signs.unsqueeze(0) * half_sizes.unsqueeze(1)) @ rotation.transpose(1, 2) + centers.unsqueeze(1)
    faces = face_template.unsqueeze(0) + 8 * torch.arange(params.shape[0], device=device).view(-1, 1, 1)
    vertex_colors = colors.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 3).clamp(0.0, 1.0)
    return vertices.reshape(-1, 3), faces.reshape(-1, 3), vertex_colors


def render_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_colors: torch.Tensor,
    *,
    camera_pose: tuple[float | None, float | None] = (25.0, -60.0),
    image_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
    grayscale: bool = False,
) -> torch.Tensor:
    """Render a mesh with PyTorch3D's differentiable rasterizer and SoftPhong shader."""
    try:
        from pytorch3d.renderer import (
            BlendParams,
            FoVPerspectiveCameras,
            MeshRasterizer,
            MeshRenderer,
            PointLights,
            RasterizationSettings,
            SoftPhongShader,
            TexturesVertex,
            look_at_view_transform,
        )
        from pytorch3d.structures import Meshes
    except ImportError as error:
        raise RuntimeError(
            "Differentiable rendering requires PyTorch3D. Install the project dependencies with `uv sync`."
        ) from error

    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if vertex_colors.shape != vertices.shape:
        raise ValueError("vertex_colors must have the same shape as vertices")
    if camera_pose[0] is None or camera_pose[1] is None:
        raise ValueError("differentiable rendering requires numeric elevation and azimuth")

    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size must contain positive dimensions")

    if grayscale:
        vertex_colors = torch.zeros_like(vertex_colors)
    mesh = Meshes(
        verts=[vertices],
        faces=[faces],
        textures=TexturesVertex(verts_features=[vertex_colors]),
    )
    rotation, translation = look_at_view_transform(
        dist=DEFAULT_CAMERA_DISTANCE,
        elev=camera_pose[0],
        azim=camera_pose[1],
        at=(DEFAULT_CAMERA_TARGET,),
        device=vertices.device,
    )
    cameras = FoVPerspectiveCameras(device=vertices.device, R=rotation, T=translation)
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=1e-4,
            faces_per_pixel=50,
            perspective_correct=True,
        ),
    )
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(
            device=vertices.device,
            cameras=cameras,
            lights=PointLights(device=vertices.device, location=((0.0, 0.0, 6.0),)),
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        ),
    )
    rendered = renderer(mesh)[0, ..., :3]
    if grayscale:
        return 1.0 - rendered.mean(dim=-1)
    return rendered


def render_cubset_differentiable(
    cubset: CubSet,
    camera_pose: tuple[float | None, float | None] = (25.0, -60.0),
    image_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
    *,
    grayscale: bool = False,
) -> torch.Tensor:
    """Create a cuboid mesh and render it through PyTorch3D."""
    vertices, faces, vertex_colors = cubset_to_mesh(cubset)
    return render_mesh(
        vertices,
        faces,
        vertex_colors,
        camera_pose=camera_pose,
        image_size=image_size,
        grayscale=grayscale,
    )


def render_cubset_differentiable_rgb_image(
    cubset: CubSet,
    camera_pose: tuple[float | None, float | None] = (25.0, -60.0),
    image_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """Render a cubset as a differentiable RGB image."""
    return render_cubset_differentiable(cubset, camera_pose, image_size)


def render_cubset_differentiable_grayscale_image(
    cubset: CubSet,
    camera_pose: tuple[float | None, float | None] = (25.0, -60.0),
    image_size: int | tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """Render a cubset as a differentiable foreground mask."""
    return render_cubset_differentiable(cubset, camera_pose, image_size, grayscale=True)
