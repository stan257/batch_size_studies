"""Catalog entries for linear and noisy linear teacher studies."""

from __future__ import annotations

from typing import List

from ..definitions import LossType, OptimizerType
from ..experiment_registry import ExperimentSpec
from ..experiment_types.synthetic import (
    SyntheticExperimentLinearTeacher,
    SyntheticExperimentNoisyLinearTeacher,
)
from .manifest import StudyManifest


def _format_linear_name(prefix: str, P: int, epochs: int, rho: float | None = None) -> str:
    base = f"{prefix}_P{P}_epochs{epochs}"
    if rho is None:
        return base
    rho_str = str(rho).replace(".", "p")
    return f"{base}_rho{rho_str}"


LINEAR_TEACHER_MANIFEST = StudyManifest(
    id="linear_teacher",
    question="How do batch size and eta interact in fixed-data linear/noisy teacher regimes?",
    family="fixed_data_linear_teacher",
    entries=(
        {"P": 2**20, "num_epochs": 1},
        {"P": 2**17, "num_epochs": 2**3},
        {"P": 2**10, "num_epochs": 2**10},
    ),
)


def build_linear_teacher_specs(manifest: StudyManifest = LINEAR_TEACHER_MANIFEST) -> List[ExperimentSpec]:
    teacher_common_kwargs = dict(
        D=500,
        alpha=2.0,
        beta=0.25,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )
    noise_levels = [0.25, 0.6]

    specs: List[ExperimentSpec] = []
    for entry in manifest.entries:
        P_val = int(entry["P"])
        epochs = int(entry["num_epochs"])
        clean_kwargs = dict(teacher_common_kwargs, P=P_val, num_epochs=epochs)
        specs.append(
            ExperimentSpec(
                name=_format_linear_name("linear_teacher", P_val, epochs),
                experiment_cls=SyntheticExperimentLinearTeacher,
                kwargs=clean_kwargs,
                family="fixed_data_linear_teacher",
                optimizer=teacher_common_kwargs["optimizer"],
                loss_type=teacher_common_kwargs["loss_type"],
            )
        )
        for rho in noise_levels:
            noisy_kwargs = dict(clean_kwargs, rho=rho)
            specs.append(
                ExperimentSpec(
                    name=_format_linear_name("noisy_linear_teacher", P_val, epochs, rho),
                    experiment_cls=SyntheticExperimentNoisyLinearTeacher,
                    kwargs=noisy_kwargs,
                    family="fixed_data_noisy_linear_teacher",
                    optimizer=teacher_common_kwargs["optimizer"],
                    loss_type=teacher_common_kwargs["loss_type"],
                )
            )
    return specs


def linear_teacher_specs() -> List[ExperimentSpec]:
    """
    Backward-compatible builder name retained for legacy imports.

    The returned specs are intentionally *not* auto-registered by import-time
    decorators. Registration now happens explicitly via studies.catalog.
    """

    return build_linear_teacher_specs()
