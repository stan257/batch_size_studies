"""Catalog entries for MNIST-1M studies."""

from __future__ import annotations

from typing import List

from ..definitions import LossType, OptimizerType, Parameterization
from ..experiment_registry import ExperimentSpec, register_spec_builder
from ..experiment_types.mnist import MNIST1MExperiment

MNIST_GAMMA_SWEEP = [0.01, 0.1, 1.0, 10.0, 100.0]


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
                        experiment_cls=MNIST1MExperiment,
                        kwargs=kwargs,
                        family="mnist1m_sampled_classification",
                        optimizer=optimizer,
                        loss_type=loss,
                    )
                )
    return specs
