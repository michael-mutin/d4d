"""Compare two cuboid OBJ meshes in an interactive window.

Run for example:
``uv run python -m tests.uc.eval.compare target.obj best_shape.obj``

When paths are omitted, file dialogs are opened. Click ``Sync views`` to copy
the current camera pose and keep both meshes synchronized while rotating.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"{path}:{line_number}: vertex needs three coordinates")
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f":
                if len(fields) < 4:
                    raise ValueError(f"{path}:{line_number}: face needs three vertices")
                face: list[int] = []
                for value in fields[1:]:
                    index = int(value.split("/", 1)[0])
                    face.append(index - 1 if index > 0 else len(vertices) + index)
                faces.extend([face[index:index + 2] + [face[0]] for index in range(1, len(face) - 1)])

    if not vertices or not faces:
        raise ValueError(f"{path}: mesh must contain vertices and faces")
    if any(index < 0 or index >= len(vertices) for face in faces for index in face):
        raise ValueError(f"{path}: face references a missing vertex")
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _draw_mesh(axis: plt.Axes, mesh: tuple[np.ndarray, np.ndarray], title: str) -> None:
    vertices, faces = mesh
    axis.set_title(title)
    axis.set_axis_off()
    axis.set_box_aspect((1, 1, 1))
    axis.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=np.asarray([face[:3] for face in faces], dtype=int),
        color="#8c5aa8",
        edgecolor="#333333",
        linewidth=0.25,
        alpha=0.9,
    )
    extent = np.ptp(vertices, axis=0).max()
    center = vertices.mean(axis=0)
    radius = max(float(extent) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _choose_mesh(title: str) -> Path:
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title=title,
        filetypes=[("Wavefront OBJ", "*.obj"), ("All files", "*.*")],
    )
    root.destroy()
    if not selected:
        raise SystemExit("No mesh selected.")
    return Path(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", type=Path, help="target OBJ mesh")
    parser.add_argument("result", nargs="?", type=Path, help="result OBJ mesh")
    args = parser.parse_args()
    target_path = args.target or _choose_mesh("Select target mesh")
    result_path = args.result or _choose_mesh("Select result mesh")

    target = _read_obj(target_path)
    result = _read_obj(result_path)
    figure = plt.figure(figsize=(12, 6))
    target_axis = figure.add_subplot(1, 2, 1, projection="3d")
    result_axis = figure.add_subplot(1, 2, 2, projection="3d")
    _draw_mesh(target_axis, target, f"Target: {target_path.name}")
    _draw_mesh(result_axis, result, f"Result: {result_path.name}")
    target_axis.view_init(elev=25, azim=-60)
    result_axis.view_init(elev=25, azim=-60)

    synced = False
    updating = False

    def copy_view(source: plt.Axes, destination: plt.Axes) -> None:
        nonlocal updating
        if updating:
            return
        updating = True
        destination.view_init(elev=source.elev, azim=source.azim, roll=source.roll)
        destination.dist = source.dist
        figure.canvas.draw_idle()
        updating = False

    def sync_button(_event) -> None:
        nonlocal synced
        synced = True
        copy_view(target_axis, result_axis)

    def on_motion(event) -> None:
        if synced and event.inaxes in (target_axis, result_axis) and event.button == 1:
            source, destination = (
                (target_axis, result_axis) if event.inaxes is target_axis else (result_axis, target_axis)
            )
            copy_view(source, destination)

    button_axis = figure.add_axes((0.42, 0.02, 0.16, 0.055))
    sync_control = Button(button_axis, "Sync views")
    sync_control.on_clicked(sync_button)
    figure.canvas.mpl_connect("motion_notify_event", on_motion)
    figure.suptitle("Target and result mesh comparison")
    figure.subplots_adjust(bottom=0.11, wspace=0.05)
    plt.show()


if __name__ == "__main__":
    main()
