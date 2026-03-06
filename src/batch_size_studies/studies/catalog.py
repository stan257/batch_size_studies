"""Aggregate study registration for default catalogs."""

from __future__ import annotations

from ..experiment_registry import list_registered_specs, register_experiment_specs
from .linear_teacher_catalog import build_linear_teacher_specs, linear_teacher_specs
from .mnist1m_catalog import (
    build_mnist1m_sampled_specs,
    build_mnist1m_specs,
    mnist1m_sampled_specs,
    mnist1m_specs,
)

DEFAULT_STUDY_BUILDERS = (
    build_linear_teacher_specs,
    build_mnist1m_specs,
    build_mnist1m_sampled_specs,
)

_REGISTERED = False


def build_default_specs():
    """Returns the full default study spec set without mutating the global registry."""
    specs = []
    for builder in DEFAULT_STUDY_BUILDERS:
        specs.extend(builder())
    return tuple(specs)


def register_default_studies(force: bool = False):
    """
    Registers all default studies exactly once unless forced.

    The function is idempotent and safe to call repeatedly from entrypoints.
    """

    global _REGISTERED

    existing = list_registered_specs()
    if _REGISTERED and existing and not force:
        return existing

    desired = build_default_specs()
    existing_names = {spec.name for spec in existing}
    specs_to_add = [spec for spec in desired if spec.name not in existing_names]

    if specs_to_add:
        register_experiment_specs(*specs_to_add)

    _REGISTERED = True
    return list_registered_specs()


# Keep import-time registration for historical behavior while exposing the
# explicit registration entrypoint above.
register_default_studies()

__all__ = [
    "DEFAULT_STUDY_BUILDERS",
    "build_default_specs",
    "register_default_studies",
    # Backward-compatible builder re-exports:
    "linear_teacher_specs",
    "mnist1m_specs",
    "mnist1m_sampled_specs",
]
