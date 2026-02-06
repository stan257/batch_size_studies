"""Catalog entries for linear and noisy linear teacher studies."""

from __future__ import annotations

from typing import List

from ..definitions import LossType, OptimizerType
from ..experiment_registry import ExperimentSpec, register_spec_builder
from ..experiment_types.synthetic import (
    SyntheticExperimentLinearTeacher,
    SyntheticExperimentNoisyLinearTeacher,
)


def _format_linear_name(prefix: str, P: int, epochs: int, rho: float | None = None) -> str:
    base = f"{prefix}_P{P}_epochs{epochs}"
    if rho is None:
        return base
    rho_str = str(rho).replace(".", "p")
    return f"{base}_rho{rho_str}"


@register_spec_builder
def linear_teacher_specs() -> List[ExperimentSpec]:
    teacher_common_kwargs = dict(
        D=500,
        alpha=2.0,
        beta=0.25,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )
    teacher_grid = [
        (2**20, 1),
        (2**17, 2**3),
        (2**10, 2**10),
    ]
    noise_levels = [0.25, 0.6]

    specs: List[ExperimentSpec] = []
    for P_val, epochs in teacher_grid:
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
