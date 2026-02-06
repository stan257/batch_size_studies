"""
Centralized Configuration for Experiments.

This module exposes canonical hyperparameter grids and materializes the
experiment objects registered via :mod:`experiment_registry`.
"""

from typing import Sequence

import numpy as np

from .definitions import LossType, OptimizerType
from .experiment_registry import ExperimentSpec, list_registered_specs

# Ensure registration side-effects run.
from .studies import catalog as _catalog  # noqa: F401  pylint: disable=unused-import


def get_main_hyperparameter_grids():
    batch_sizes = (2 ** np.arange(0, 17)).tolist()
    etas = np.power(2.0, np.arange(-12, 14)).tolist()
    return batch_sizes, etas


def _filter_specs(
    specs: Sequence[ExperimentSpec],
    experiment_types: Sequence[str] | None,
    optimizer: OptimizerType | None,
    loss_type: LossType | None,
):
    type_filter = set(experiment_types) if experiment_types is not None else None
    for spec in specs:
        if spec.matches(type_filter, optimizer=optimizer, loss_type=loss_type):
            yield spec


def get_main_experiment_configs(
    experiment_types: Sequence[str] | None = None,
    optimizer: OptimizerType | None = None,
    loss_type: LossType | None = None,
):
    experiments_to_run = {}
    specs = list_registered_specs()
    for spec in _filter_specs(specs, experiment_types, optimizer, loss_type):
        experiments_to_run[spec.name] = spec.build()
    return experiments_to_run


def list_main_experiment_specs():
    return list_registered_specs()
