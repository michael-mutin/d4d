"""Rewrite data types for the union-cuboid grammar."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitX:
    i: int
    t: float


@dataclass(frozen=True)
class SplitY:
    i: int
    t: float


@dataclass(frozen=True)
class SplitZ:
    i: int
    t: float


@dataclass(frozen=True)
class AddCub:
    x: float
    y: float
    z: float
    s: float


@dataclass(frozen=True)
class AddHoleXY:
    i: int
    x: float
    y: float
    s: float


@dataclass(frozen=True)
class AddHoleXZ:
    i: int
    x: float
    z: float
    s: float


@dataclass(frozen=True)
class AddHoleYZ:
    i: int
    y: float
    z: float
    s: float


@dataclass(frozen=True)
class RemoveRect:
    i: int


@dataclass(frozen=True)
class Merge:
    i: int
    j: int


Rewrite = SplitX | SplitY | SplitZ | AddCub | AddHoleXY | AddHoleXZ | AddHoleYZ | RemoveRect | Merge
