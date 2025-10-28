from __future__ import annotations

import logging
import os
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum

import jax.random as jr
import numpy as np

from .definitions import LossType, OptimizerType, Parameterization
from .models import MLP
from .storage_utils import generate_experiment_filename, load_experiment, save_experiment


# Student Mixins
@dataclass(frozen=True)
class MLPStudentExperiment:
    """A mixin for experiments that use an MLP as the student model."""

    N: int
    L: int
    parameterization: Parameterization
    gamma: float


@dataclass(frozen=True)
class LinearStudentExperiment:
    """A mixin for experiments that use a LinearModel as the student."""

    D: int


# Base class for all experiments
@dataclass(frozen=True)
class ExperimentBase:
    """
    A base class that provides file I/O and automatic type validation for experiment dataclasses.
    """

    optimizer: OptimizerType
    loss_type: LossType
    experiment_type: str = field(init=False)

    def __post_init__(self):
        """
        Performs strict type checking on all attributes after initialization.
        """
        resolved_types = typing.get_type_hints(self.__class__)
        for field_name, field_def in self.__class__.__dataclass_fields__.items():
            if not field_def.init:
                continue
            value = getattr(self, field_name)
            expected_type = resolved_types[field_name]
            origin = typing.get_origin(expected_type)
            union_types = (typing.Union,)
            if hasattr(typing, "UnionType"):
                union_types += (typing.UnionType,)
            if origin in union_types:
                args = typing.get_args(expected_type)
                if not isinstance(value, args):
                    raise TypeError(
                        f"Attribute '{field_name}' expected one of types {args}, but got {type(value).__name__}."
                    )
            elif origin is None:
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Attribute '{field_name}' expected type {expected_type.__name__}, "
                        f"but got {type(value).__name__}."
                    )

    def to_params_dict(self):
        """
        Helper to return dataclass attributes as a dictionary, converting Enums
        to their string values and excluding 'experiment_type' for cleaner filenames. The
        dictionary is sorted by key to ensure consistent filename ordering.
        """
        params = asdict(self)
        params.pop("num_outputs", None)  # Exclude constants from filename
        params.pop("experiment_type", None)
        for key, value in params.items():
            if isinstance(value, Enum):
                params[key] = value.value

        # Sort the dictionary by key for consistent filenames
        return dict(sorted(params.items()))

    def generate_filename(self, prefix="results", extension="pkl"):
        params = self.to_params_dict()
        return generate_experiment_filename(params, prefix, extension)

    def get_filepath(self, directory="experiments", prefix="results", extension="pkl"):
        type_specific_directory = os.path.join(directory, self.experiment_type)
        filename = self.generate_filename(prefix, extension)
        return os.path.join(type_specific_directory, filename)

    def load_results(self, directory="experiments", prefix="results", extension="pkl", silent: bool = False):
        filepath = self.get_filepath(directory, prefix, extension)
        data = load_experiment(filepath)
        if data:
            if not silent:
                logging.info(f"Results file found, loading from: {os.path.basename(filepath)}")
            losses = defaultdict(list, data.get("losses", {}))
            failed_runs = data.get("failed_runs", set())
            return losses, failed_runs
        else:
            if not silent:
                logging.info("No results file found for this experiment. Initializing new results.")
            return defaultdict(list), set()

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


class SyntheticExperiment(ABC):
    @abstractmethod
    def generate_teacher_weights(self): ...

    @abstractmethod
    def generate_data(self, data_key): ...


@dataclass(frozen=True)
class SyntheticExperimentFixedTime(ExperimentBase, MLPStudentExperiment, SyntheticExperiment):
    # Student and task parameters
    D: int
    P: int
    K: int
    num_steps: int
    experiment_type: str = field(default="fixed_time_poly_teacher", init=False)

    def __post_init__(self):
        super().__post_init__()

    def generate_teacher_weights(self):
        key = jr.key(0)
        return jr.normal(key, (self.D, 1)) / np.sqrt(self.D)

    def generate_data(self, data_key):
        p, d, k, w = self.P, self.D, self.K, self.generate_teacher_weights()
        X_key, _ = jr.split(data_key, 2)
        X_data = jr.normal(X_key, (p, d))
        y_data = (X_data @ w) ** k
        return X_data, y_data

    def plot_title(self, task_name="poly task", model_name="MLP"):
        line1 = f"$T* = {self.num_steps}$ steps, {task_name} w/ $k={self.K}, D={self.D}$"
        line2 = f"{model_name} in {self.parameterization.value} w/ $N={self.N}, L={self.L}, \\gamma={self.gamma}$"
        return f"{line1}\n{line2}"


@dataclass(frozen=True)
class SyntheticExperimentFixedData(ExperimentBase, MLPStudentExperiment, SyntheticExperiment):
    # Student and task parameters
    D: int
    P: int
    K: int
    seed: int = 0  # Seed for reproducible data generation
    experiment_type: str = field(default="fixed_data_poly_teacher", init=False)

    def __post_init__(self):
        super().__post_init__()

    def generate_teacher_weights(self):
        key = jr.key(0)
        return jr.normal(key, (self.D, 1)) / np.sqrt(self.D)

    def generate_data(self, data_key):
        p, d, k, w = self.P, self.D, self.K, self.generate_teacher_weights()
        X_key, _ = jr.split(data_key, 2)
        X_data = jr.normal(X_key, (p, d))
        y_data = (X_data @ w) ** k
        return X_data, y_data

    def plot_title(self, task_name="poly task", model_name="MLP"):
        line1 = f"$P = {self.P}$ samples, {task_name} w/ $k={self.K}, D={self.D}$"
        line2 = f"{model_name} in {self.parameterization.value} w/ $N={self.N}, L={self.L}, \\gamma={self.gamma}$"
        return f"{line1}\n{line2}"


# TODO: Make separate MLP fixed-data class for experiments, when needed
@dataclass(frozen=True)
class SyntheticExperimentMLPTeacher(ExperimentBase, MLPStudentExperiment, SyntheticExperiment):
    """
    Defines parameters for a synthetic data experiment where the teacher is an MLP.
    This is a *fixed-time* experiment.
    """

    # Additional student parameters
    D: int
    P: int
    num_steps: int

    # Teacher parameters
    teacher_N: int
    teacher_L: int
    teacher_gamma: float
    teacher_parameterization: Parameterization

    experiment_type: str = field(default="fixed_time_mlp_teacher", init=False)

    def __post_init__(self):
        super().__post_init__()

    def generate_teacher_weights(self):
        teacher_model = MLP(parameterization=self.teacher_parameterization, gamma=self.teacher_gamma)
        teacher_widths = [self.D] + [self.teacher_N] * (self.teacher_L - 1) + [1]
        # Use a fixed key for a deterministic teacher
        return teacher_model.init_params(init_key=1, widths=teacher_widths)

    def generate_data(self, data_key):
        """Generates a synthetic dataset using the MLP teacher."""
        teacher_weights = self.generate_teacher_weights()
        teacher_model = MLP(parameterization=self.teacher_parameterization, gamma=self.teacher_gamma)

        X_key, _ = jr.split(data_key, 2)
        X_data = jr.normal(X_key, (self.P, self.D))

        y_data = teacher_model(teacher_weights, X_data)

        return X_data, y_data

    def plot_title(self, task_name="MLP teacher", model_name="MLP"):
        line1 = f"$T* = {self.num_steps}$ steps, {task_name} T(N={self.teacher_N}, L={self.teacher_L}), D={self.D}"
        line2 = f"{model_name} in {self.parameterization.value} w/ $N={self.N}, L={self.L}, \\gamma={self.gamma}$"
        return f"{line1}\n{line2}"


@dataclass(frozen=True)
class SyntheticExperimentLinearTeacher(ExperimentBase, LinearStudentExperiment, SyntheticExperiment):
    """
    Defines parameters for a synthetic data experiment where the teacher is linear.
    This is a *fixed-data* experiment.
    y = w^T x
    x ~ N(0, Sigma), Sigma = diag(i^-alpha for i=1..D)
    w_i = i^(-theta), where theta = 1/2 + 1/2 * alpha * (beta - 1)

    Here alpha and beta are the capacity and source exponents as defined in
    https://abatanasov.com/Files/Scaling_Laws_Note.pdf, i.e. the decay rate of the
    data-covariance and of the tail of Var(y), respectively.
    """

    # Student and task parameters
    P: int

    # Teacher parameters
    alpha: float
    beta: float

    seed: int = 0  # Seed for reproducible data generation
    experiment_type: str = field(default="fixed_data_linear_teacher", init=False)

    def __post_init__(self):
        super().__post_init__()

    def generate_teacher_weights(self):
        # w_i = i^(-theta)
        indices = np.arange(1, self.D + 1)
        theta = 1 / 2 + 1 / 2 * self.alpha * (self.beta - 1)
        w = indices ** (-theta)
        return w.reshape(-1, 1)

    def generate_data(self, data_key):
        # x ~ N(0, Sigma), Sigma = diag(i^-alpha for i=1..D)
        # w_i = i^(-theta)
        # y = X @ w
        w = self.generate_teacher_weights()

        # Generate standard normal data∂
        z_key, _ = jr.split(data_key, 2)
        z_data = jr.normal(z_key, (self.P, self.D))

        # Scale by sqrt(Sigma)
        # sqrt(Sigma) is a diagonal matrix with sqrt of Sigma's diagonal elements
        sigma_diag_sqrt = np.arange(1, self.D + 1) ** (-self.alpha / 2.0)
        X_data = z_data * sigma_diag_sqrt  # Element-wise multiplication, broadcasting

        y_data = X_data @ w
        return X_data, y_data

    def plot_title(self, task_name="linear task", model_name="Linear Model"):
        line1 = f"$P = {self.P}$ samples, {task_name} w/ $D={self.D}, \\alpha={self.alpha}, \\beta={self.beta}$"
        line2 = f"Student: {model_name}, Optimizer: {self.optimizer.value}, Loss: {self.loss_type.value}"
        return f"{line1}\n{line2}"


@dataclass(frozen=True)
class MNISTExperiment(ExperimentBase, MLPStudentExperiment):
    """
    Defines parameters for a 10-class MNIST classification experiment.
    This is a fixed-data, fixed-epoch experiment that uses the custom MLP
    from models.py. Sweeps are performed over batch size and learning rate.
    """

    D: int = field(default=784, init=False)  # Input dim for flattened MNIST
    num_outputs: int = 10

    # Training parameters
    num_epochs: int = 1

    experiment_type: str = field(default="mnist_classification", init=False)

    def __post_init__(self):
        super().__post_init__()

    def plot_title(self, task_name="MNIST Classification", model_name="MLP"):
        line1 = (
            f"{task_name} ({model_name} N={self.N}, L={self.L}, {self.parameterization.value}, $\\gamma={self.gamma}$)"
        )
        line2 = f"Epochs={self.num_epochs}, Optimizer={self.optimizer.value}"
        return f"{line1}\n{line2}"


@dataclass(frozen=True)
class MNIST1MExperiment(ExperimentBase, MLPStudentExperiment):
    """
    Defines parameters for a 10-class MNIST-1M classification experiment.
    This dataset is generated from a diffusion model.
    """

    num_epochs: int
    D: int = field(default=784, init=False)
    num_outputs: int = 10

    experiment_type: str = field(default="mnist1m_classification", init=False)

    def __post_init__(self):
        super().__post_init__()

    def plot_title(self, task_name="MNIST-1M Classification", model_name="MLP"):
        line1 = (
            f"{task_name} ({model_name} N={self.N}, L={self.L}, {self.parameterization.value}, $\\gamma={self.gamma}$)"
        )
        line2 = f"Epochs={self.num_epochs}, Optimizer={self.optimizer.value}, Loss={self.loss_type.value}"
        return f"{line1}\n{line2}"


@dataclass(frozen=True)
class MNIST1MSampledExperiment(ExperimentBase, MLPStudentExperiment):
    """
    An MNIST-1M experiment that trains on a subset of the full dataset.
    This is useful for quick sanity checks and faster experimental cycles.
    """

    num_epochs: int
    max_train_samples: int
    num_outputs: int = 10
    D: int = field(default=784, init=False)
    experiment_type: str = field(default="mnist1m_sampled_classification", init=False)

    def __post_init__(self):
        super().__post_init__()

    def plot_title(self, task_name="MNIST-1M Sampled", model_name="MLP"):
        line1 = (
            f"{task_name} ({model_name} N={self.N}, L={self.L}, {self.parameterization.value}, $\\gamma={self.gamma}$)"
        )
        line2 = f"Epochs={self.num_epochs}, Optimizer={self.optimizer.value}, Loss={self.loss_type.value}, Samples={self.max_train_samples}"
        return f"{line1}\n{line2}"
