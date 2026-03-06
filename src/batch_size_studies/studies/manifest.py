"""Declarative study manifest primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StudyManifest:
    """
    Human-readable study declaration.

    The manifest is intentionally lightweight: it carries descriptive metadata and
    a list of concrete grid entries used to instantiate :class:`ExperimentSpec`.
    """

    id: str
    question: str
    family: str
    entries: tuple[dict[str, Any], ...]
