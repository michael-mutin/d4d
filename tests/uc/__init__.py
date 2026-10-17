"""Parametric union-cuboid shape grammar."""

from .grammar import CubGrammar
from .object import CubSet
from .rendering import render_cubset
from .rewrite import (
    AddCub,
    AddHoleXY,
    AddHoleXZ,
    AddHoleYZ,
    Merge,
    RemoveRect,
    Rewrite,
    SplitX,
    SplitY,
    SplitZ,
)

__all__ = [
    "AddCub",
    "AddHoleXY",
    "AddHoleXZ",
    "AddHoleYZ",
    "CubGrammar",
    "CubSet",
    "Merge",
    "RemoveRect",
    "Rewrite",
    "SplitX",
    "SplitY",
    "SplitZ",
    "render_cubset",
]
