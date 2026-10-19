"""Parametric union-cuboid shape grammar."""

from .uc_grammar import CubGrammar
from .uc_object import CubSet
from .uc_rendering import render_cubset
from .uc_rewrite import (
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
