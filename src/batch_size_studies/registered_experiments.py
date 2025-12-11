"""
This module contains the declarative specifications for all experiment sweeps.
It acts as the central catalog for the framework.

To add a new experiment, follow these steps:
1. Define your experiment configuration as a dataclass in `experiments.py`
   (or a new module if it's a new family of experiments).
2. Create a builder function here (e.g., `_build_my_new_experiment`) that
   returns a list of `ExperimentSpec` objects.
3. Decorate the builder function with `@register_spec_builder`.

The experiment will then be automatically available to the CLI and notebooks.
"""

from __future__ import annotations

from typing import List

from .definitions import LossType, OptimizerType, Parameterization
from .experiment_registry import ExperimentSpec, register_spec_builder
from .experiment_types.mnist import MNIST1MExperiment, MNIST1MSampledExperiment
from .experiment_types.synthetic import (
    SyntheticExperimentLinearTeacher,
    SyntheticExperimentNoisyLinearTeacher,
)

MNIST_GAMMA_SWEEP = [0.01, 0.1, 1.0, 10.0, 100.0]


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


@register_spec_builder
def mnist1m_specs() -> List[ExperimentSpec]:
    base_kwargs = dict(
        N=128,
        L=3,
        parameterization=Parameterization.MUP,
        num_epochs=1,
    )
    specs: List[ExperimentSpec] = []
    for optimizer in OptimizerType:
        for loss in LossType:
            for gamma in MNIST_GAMMA_SWEEP:
                name = (
                    f"mnist1m_mup_{loss.value}_{optimizer.value}_gamma"
                    f"{str(gamma).replace('.', 'p')}_epochs{base_kwargs['num_epochs']}"
                )
                kwargs = dict(base_kwargs, optimizer=optimizer, loss_type=loss, gamma=gamma)
                specs.append(
                    ExperimentSpec(
                        name=name,
                        experiment_cls=MNIST1MExperiment,
                        kwargs=kwargs,
                        family="mnist1m_classification",
                        optimizer=optimizer,
                        loss_type=loss,
                    )
                )
    return specs


@register_spec_builder
def mnist1m_sampled_specs() -> List[ExperimentSpec]:
    base_kwargs = dict(
        N=128,
        L=3,
        parameterization=Parameterization.MUP,
        num_epochs=20,
        max_train_samples=65_536,
    )
    specs: List[ExperimentSpec] = []
    for optimizer in OptimizerType:
        for loss in LossType:
            for gamma in MNIST_GAMMA_SWEEP:
                name = (
                    f"mnist1m_sampled_mup_{loss.value}_{optimizer.value}_gamma"
                    f"{str(gamma).replace('.', 'p')}_epochs{base_kwargs['num_epochs']}"
                )
                kwargs = dict(base_kwargs, optimizer=optimizer, loss_type=loss, gamma=gamma)
                specs.append(
                    ExperimentSpec(
                        name=name,
                        experiment_cls=MNIST1MSampledExperiment,
                        kwargs=kwargs,
                        family="mnist1m_sampled_classification",
                        optimizer=optimizer,
                        loss_type=loss,
                    )
                )
    return specs
