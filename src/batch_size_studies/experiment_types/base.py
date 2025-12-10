from __future__ import annotations

import itertools
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type

import numpy as np

from ..definitions import LossType, OptimizerType, Parameterization
from ..models import MLP, LinearModel
from ..protocols import TrialRunner
from ..storage_utils import generate_experiment_filename, load_experiment, save_experiment


@dataclass(frozen=True)
class MLPStudentExperiment:
    """Mixin for experiments that use an MLP as the student model."""

    N: int
    L: int
    parameterization: Parameterization
    gamma: float

    def create_model_instance(self):
        return MLP(parameterization=self.parameterization, gamma=self.gamma)

    def get_output_dim(self) -> int:
        return 1

    def get_model_widths(self) -> list[int]:
        return [self.D] + [self.N] * (self.L - 1) + [self.get_output_dim()]

    def get_model_wrapper(self, model_instance, params0):
        from ..models import CenteredModel

        return CenteredModel(model_instance, params0)

    def get_adjusted_eta(self, base_eta: float) -> float:
        gamma = self.gamma
        depth = self.L
        width = self.N

        match self.optimizer:
            case OptimizerType.SGD:
                gamma_mult = gamma ** (2 / depth) if gamma > 1 else gamma**2
            case OptimizerType.ADAM:
                gamma_mult = gamma ** (1 / depth) if gamma > 1 else gamma
            case _:
                gamma_mult = 1.0

        if self.parameterization == Parameterization.SP:
            width_mult = 1.0
        else:
            match self.optimizer:
                case OptimizerType.SGD:
                    width_mult = width
                case OptimizerType.ADAM:
                    width_mult = np.sqrt(width)
                case _:
                    width_mult = 1.0

        return base_eta * gamma_mult * width_mult


@dataclass(frozen=True)
class LinearStudentExperiment:
    """Mixin for experiments that use a LinearModel."""

    D: int

    def create_model_instance(self):
        return LinearModel()

    def get_model_widths(self) -> list[int]:
        return [self.D, 1]

    def get_model_wrapper(self, model_instance, params0):
        return model_instance

    def get_adjusted_eta(self, base_eta: float) -> float:
        return base_eta


@dataclass(frozen=True)
class ExperimentBase(ABC):
    """Base class that provides common sweep/IO utilities."""

    optimizer: OptimizerType
    loss_type: LossType
    experiment_type: str = field(init=False)

    def to_params_dict(self):
        params = asdict(self)
        params.pop("num_outputs", None)
        params.pop("experiment_type", None)
        for key, value in params.items():
            if isinstance(value, Enum):
                params[key] = value.value
        return dict(sorted(params.items()))

    def _filename_optional_params(self) -> tuple[str, ...]:
        return ("loss_type",)

    def get_filename_variants(self, prefix="results", extension="pkl") -> list[str]:
        params = self.to_params_dict()
        variants: list[str] = []
        seen: set[str] = set()

        def add_variant(active_params: dict):
            filename = generate_experiment_filename(active_params, prefix, extension)
            if filename not in seen:
                variants.append(filename)
                seen.add(filename)

        add_variant(params)
        optional_keys = [key for key in self._filename_optional_params() if key in params]
        for r in range(1, len(optional_keys) + 1):
            for combo in itertools.combinations(optional_keys, r):
                variant_params = {k: v for k, v in params.items() if k not in combo}
                add_variant(variant_params)
        return variants

    def generate_filename(self, prefix="results", extension="pkl"):
        return self.get_filename_variants(prefix, extension)[0]

    def get_filepath(self, directory="experiments", prefix="results", extension="pkl"):
        type_specific_directory = os.path.join(directory, self.experiment_type)
        filename = self.generate_filename(prefix, extension)
        return os.path.join(type_specific_directory, filename)

    def load_results(self, directory="experiments", prefix="results", extension="pkl", silent: bool = False):
        type_specific_directory = os.path.join(directory, self.experiment_type)
        filepath = None
        for filename in self.get_filename_variants(prefix, extension):
            candidate = os.path.join(type_specific_directory, filename)
            if os.path.exists(candidate):
                filepath = candidate
                break
        data = load_experiment(filepath) if filepath is not None else None
        if data:
            if not silent:
                logging.info(f"Results file found, loading from: {os.path.basename(filepath)}")
            losses = data.get("losses", {})
            failed_runs = data.get("failed_runs", set())
            return losses, failed_runs
        if not silent:
            logging.info("No results file found for this experiment. Initializing new results.")
        return {}, set()

    def save_results(
        self,
        losses: dict,
        failed_runs: set,
        directory="experiments",
        prefix="results",
        extension="pkl",
    ):
        data_to_save = {"losses": losses, "failed_runs": failed_runs}
        filepath = self.get_filepath(directory, prefix, extension)
        return save_experiment(data_to_save, filepath)

    def describe(self):
        pprint(self)

    @abstractmethod
    def should_skip_batch_size(self, batch_size: int, train_ds: any | None = None) -> bool: ...

    @abstractmethod
    def get_trial_runner_class(self) -> Type["TrialRunner"]: ...

    def get_sweep_metadata(self, init_key: int) -> dict:
        return {}

    @abstractmethod
    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]: ...

    def get_default_dataset_loader(self):
        """Optional hook for experiments to provide a default dataset loader callable."""
        return None

    @abstractmethod
    def get_model_widths(self) -> list[int]: ...

    @abstractmethod
    def get_model_wrapper(self, model_instance, params0): ...

    @abstractmethod
    def get_adjusted_eta(self, base_eta: float) -> float: ...

    @abstractmethod
    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]: ...

    def is_online_experiment(self) -> bool:
        return False

    def is_classification(self) -> bool:
        return False
