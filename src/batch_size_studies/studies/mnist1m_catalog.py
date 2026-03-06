"""Catalog entries for MNIST-1M studies."""

from __future__ import annotations

from typing import List

from ..definitions import LossType, OptimizerType, Parameterization
from ..experiment_registry import ExperimentSpec
from ..experiment_types.mnist import MNIST1MExperiment
from .manifest import StudyManifest

MNIST_GAMMA_SWEEP = [0.01, 0.1, 1.0, 10.0, 100.0]


MNIST1M_FULL_MANIFEST = StudyManifest(
    id="mnist1m_full",
    question="How do eta and batch size affect full MNIST-1M training dynamics?",
    family="mnist1m_classification",
    entries=({"N": 128, "L": 3, "parameterization": Parameterization.MUP, "num_epochs": 1, "max_train_samples": None},),
)

MNIST1M_SAMPLED_MANIFEST = StudyManifest(
    id="mnist1m_sampled",
    question="How do eta and batch size behave under sampled MNIST-1M regimes?",
    family="mnist1m_sampled_classification",
    entries=(
        {"N": 128, "L": 3, "parameterization": Parameterization.MUP, "num_epochs": 20, "max_train_samples": 65_536},
    ),
)


def build_mnist1m_specs(manifest: StudyManifest = MNIST1M_FULL_MANIFEST) -> List[ExperimentSpec]:
    if len(manifest.entries) != 1:
        raise ValueError("MNIST1M full manifest expects exactly one base entry.")
    base_kwargs = dict(manifest.entries[0])
    base_kwargs.pop("max_train_samples", None)
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


def build_mnist1m_sampled_specs(manifest: StudyManifest = MNIST1M_SAMPLED_MANIFEST) -> List[ExperimentSpec]:
    if len(manifest.entries) != 1:
        raise ValueError("MNIST1M sampled manifest expects exactly one base entry.")
    base_kwargs = dict(manifest.entries[0])
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


def mnist1m_specs() -> List[ExperimentSpec]:
    """
    Backward-compatible builder retained for legacy imports.

    Registration is now explicit via studies.catalog.
    """

    return build_mnist1m_specs()


def mnist1m_sampled_specs() -> List[ExperimentSpec]:
    """
    Backward-compatible builder retained for legacy imports.

    Registration is now explicit via studies.catalog.
    """

    return build_mnist1m_sampled_specs()
