from __future__ import annotations

import itertools
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.flatten_util import ravel_pytree

from .constants import SYNTH_EVAL_DATA_SEED_OFFSET, SYNTH_EVAL_MAX_SAMPLES, SYNTH_EVAL_SUBSET_SEED_OFFSET
from .data_loading import load_datasets, load_mnist1m_dataset
from .definitions import LossType, OptimizerType, Parameterization
from .models import MLP, LinearModel
from .protocols import TrialRunner
from .storage_utils import generate_experiment_filename, load_experiment, save_experiment


def _subsample_mnist_data(train_images, train_labels, experiment, init_key):
    """Helper to subsample MNIST training data."""
    num_samples_to_use = getattr(experiment, "max_train_samples", None)
    if num_samples_to_use is not None and num_samples_to_use > 0:
        num_original_samples = len(train_images)
        if num_original_samples >= num_samples_to_use:
            shuffle_key = jr.PRNGKey(init_key)
            indices_to_use = jr.permutation(shuffle_key, num_original_samples)[:num_samples_to_use]
            train_images = train_images[np.array(indices_to_use)]
            train_labels = train_labels[np.array(indices_to_use)]
            logging.info(f"Training on a random subset of {len(train_images)} samples.")
    return train_images, train_labels


def _load_mnist_dataset(experiment, init_key: int, dataset_loader=None, forced_subsample_seed=None):
    """Loads MNIST dataset with optional subsampling."""
    if dataset_loader is None:
        dataset_loader = load_datasets if isinstance(experiment, MNISTExperiment) else load_mnist1m_dataset

    try:
        (train_images, train_labels), (test_images, test_labels) = dataset_loader()
        if isinstance(experiment, MNIST1MSampledExperiment):
            # Use the forced seed if provided (for analysis tools), otherwise use the runtime init_key.
            seed_to_use = forced_subsample_seed if forced_subsample_seed is not None else init_key
            train_images, train_labels = _subsample_mnist_data(train_images, train_labels, experiment, seed_to_use)
        train_ds = {"image": train_images, "label": train_labels}
        test_ds = {"image": test_images, "label": test_labels}
        return train_ds, test_ds
    except Exception as e:
        logging.error(f"Failed to load dataset: {type(e).__name__}: {e}")
        return None, None


# Student Mixins
@dataclass(frozen=True)
class MLPStudentExperiment:
    """A mixin for experiments that use an MLP as the student model."""

    N: int
    L: int
    parameterization: Parameterization
    gamma: float

    def create_model_instance(self):
        """Creates an MLP model instance based on the experiment's configuration."""
        return MLP(parameterization=self.parameterization, gamma=self.gamma)

    def get_output_dim(self) -> int:
        """Returns the output dimension of the model. Override in subclasses for multi-class tasks."""
        return 1

    def get_model_widths(self) -> list[int]:
        """Computes the layer widths for the MLP model."""
        return [self.D] + [self.N] * (self.L - 1) + [self.get_output_dim()]

    def get_model_wrapper(self, model_instance, params0):
        """Wraps the MLP model instance with CenteredModel."""
        from .models import CenteredModel

        return CenteredModel(model_instance, params0)

    def get_adjusted_eta(self, base_eta: float) -> float:
        """
        The learning rate adjustment schedule is based on the findings from:
        "The Optimization Landscape of SGD Across the Feature Learning Strength"
        Atanasov et al. (2025, arXiv:2410.04642)

        For μP, we scale by the width N for SGD (resp. sqrt(N) for ADAM) to ensure
        μ-transfer across width.
        """
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
        else:  # muP
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
    """A mixin for experiments that use a LinearModel as the student."""

    D: int

    def create_model_instance(self):
        """Creates a LinearModel instance."""
        return LinearModel()

    def get_model_widths(self) -> list[int]:
        """Computes the layer widths for the Linear model."""
        # Output dimension is always 1 for linear models in this framework.
        return [self.D, 1]

    def get_model_wrapper(self, model_instance, params0):
        """Returns the raw Linear model instance without a wrapper."""
        return model_instance

    def get_adjusted_eta(self, base_eta: float) -> float:
        """Returns the unadjusted learning rate for linear models."""
        return base_eta


# Base class for all experiments
@dataclass(frozen=True)
class ExperimentBase(ABC):
    """
    A base class that provides file I/O and automatic type validation for experiment dataclasses.
    """

    optimizer: OptimizerType
    loss_type: LossType
    experiment_type: str = field(init=False)

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

    def _filename_optional_params(self) -> tuple[str, ...]:
        """Returns params that may be absent in legacy filenames."""
        return ("loss_type",)

    def get_filename_variants(self, prefix="results", extension="pkl") -> list[str]:
        """
        Returns possible filename variants (new first, legacy fallbacks after).
        """
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

        if filepath is not None:
            data = load_experiment(filepath)
        else:
            data = None

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
        """Pretty-prints the experiment configuration for easy inspection."""
        pprint(self)

    @abstractmethod
    def should_skip_batch_size(self, batch_size: int, train_ds: any | None = None) -> bool:
        """
        Checks if a given batch size is invalid for this experiment.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trial_runner_class(self) -> Type["TrialRunner"]:
        """
        Returns the specific TrialRunner class required for this experiment.
        """
        raise NotImplementedError

    def get_sweep_metadata(self, init_key: int) -> dict:
        """Returns sweep-level metadata to be saved, e.g., seeds for data subsampling."""
        return {}

    @abstractmethod
    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        """
        Prepares training and test datasets for the experiment.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_widths(self) -> list[int]:
        """
        Returns the list of layer widths for the model architecture.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_wrapper(self, model_instance, params0):
        """
        Wraps the model instance if necessary (e.g., for centering).
        """
        raise NotImplementedError

    @abstractmethod
    def get_adjusted_eta(self, base_eta: float) -> float:
        """
        Returns the final, adjusted learning rate for the optimizer.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        """
        Computes the total number of training steps and the effective number of epochs for a trial.
        Returns: (num_steps, effective_num_epochs)
        """
        raise NotImplementedError

    def is_online_experiment(self) -> bool:
        """
        Returns True if the experiment does not require a pre-loaded dataset.
        This is used by the runner to decide if it should abort on data loading failure.
        """
        return False

    def is_classification(self) -> bool:
        """
        Returns True if the experiment targets discrete labels (e.g., MNIST variants).
        Defaults to False; classification experiments should override.
        """
        return False


class SyntheticExperiment(ABC):
    @abstractmethod
    def generate_teacher_weights(self): ...

    @abstractmethod
    def generate_data(self, data_key): ...


@dataclass(frozen=True)
class SyntheticExperimentFixedTime(MLPStudentExperiment, ExperimentBase, SyntheticExperiment):
    """
    MRO: SyntheticExperimentFixedTime -> MLPStudentExperiment -> ExperimentBase -> SyntheticExperiment -> object
    Model configuration comes from MLPStudentExperiment.
    Infrastructure (I/O, validation) comes from ExperimentBase.
    """

    # Student and task parameters
    D: int
    P: int
    K: int
    num_steps: int
    experiment_type: str = field(default="fixed_time_poly_teacher", init=False)

    def __post_init__(self):
        if self.D <= 0:
            raise ValueError(f"D must be positive, got {self.D}")
        if self.P <= 0:
            raise ValueError(f"P must be positive, got {self.P}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

    def generate_teacher_weights(self):
        key = jr.PRNGKey(0)
        return jr.normal(key, (self.D, 1)) / np.sqrt(self.D)

    def generate_data(self, data_key):
        p, d, k, w = self.P, self.D, self.K, self.generate_teacher_weights()
        X_key, _ = jr.split(data_key, 2)
        X_data = jr.normal(X_key, (p, d))
        y_data = (X_data @ w) ** k
        return X_data, y_data

    def plot_title(self, task_name="poly task", model_name="MLP"):
        line1 = f"$T* = {self.num_steps}$ steps, {task_name} (Online) w/ $k={self.K}, D={self.D}$"
        line2 = f"{model_name} in {self.parameterization.value} w/ $N={self.N}, L={self.L}, \\gamma={self.gamma}$"
        return f"{line1}\n{line2}"

    def should_skip_batch_size(self, batch_size: int, train_ds: any | None = None) -> bool:
        # Even though data is regenerated online, batches larger than P yield no data.
        if batch_size <= 0 or batch_size > self.P:
            logging.warning(f"Skipping batch size {batch_size} > synthetic block size P ({self.P}).")
            return True
        return False

    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        # For fixed-time experiments, num_epochs is not relevant. We can consider it as 1.
        return self.num_steps, 1

    def is_online_experiment(self) -> bool:
        return True

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        """Returns the runner for fixed-time synthetic experiments."""
        from .trainer import SyntheticFixedTimeTrialRunner

        return SyntheticFixedTimeTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        """Fixed-time experiments do not load a dataset upfront."""
        return None, None


@dataclass(frozen=True)
class SyntheticExperimentFixedData(MLPStudentExperiment, ExperimentBase, SyntheticExperiment):
    """
    MRO: SyntheticExperimentFixedData -> MLPStudentExperiment -> ExperimentBase -> SyntheticExperiment -> object
    Model configuration comes from MLPStudentExperiment.
    Infrastructure (I/O, validation) comes from ExperimentBase.
    """

    # Student and task parameters
    D: int
    P: int
    K: int
    num_epochs: int = 1
    seed: int = 0  # Seed for reproducible data generation
    experiment_type: str = field(default="fixed_data_poly_teacher", init=False)

    def __post_init__(self):
        if self.D <= 0:
            raise ValueError(f"D must be positive, got {self.D}")
        if self.P <= 0:
            raise ValueError(f"P must be positive, got {self.P}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

    def generate_teacher_weights(self):
        key = jr.PRNGKey(0)
        return jr.normal(key, (self.D, 1)) / np.sqrt(self.D)

    def generate_data(self, data_key):
        p, d, k, w = self.P, self.D, self.K, self.generate_teacher_weights()
        X_key, _ = jr.split(data_key, 2)
        X_data = jr.normal(X_key, (p, d))
        y_data = (X_data @ w) ** k
        return X_data, y_data

    def plot_title(self, task_name="poly task", model_name="MLP"):
        num_epochs = getattr(self, "num_epochs", 1)
        learning_type = "Online" if num_epochs == 1 else "Offline"
        line1 = f"$P = {self.P}$ samples, {task_name} ({learning_type}) w/ $k={self.K}, D={self.D}$"
        line2 = f"{model_name} in {self.parameterization.value} w/ $N={self.N}, L={self.L}, \\gamma={self.gamma}$"
        if num_epochs > 1:
            line2 += f", Epochs={num_epochs}"
        return f"{line1}\n{line2}"

    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        epochs_to_run = num_epochs if num_epochs is not None else self.num_epochs
        steps_per_epoch = self.P // batch_size
        return epochs_to_run * steps_per_epoch, epochs_to_run

    def should_skip_batch_size(self, batch_size: int, train_ds: any | None = None) -> bool:
        # P is the authoritative dataset size for this experiment type.
        if batch_size <= 0 or batch_size > self.P:
            logging.warning(f"Skipping batch size {batch_size} > dataset size P ({self.P}).")
            return True
        return False

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        """Returns the runner for fixed-data synthetic experiments."""
        from .trainer import SyntheticFixedDataTrialRunner

        return SyntheticFixedDataTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        """Generates the synthetic dataset for this experiment."""
        data_key = jr.PRNGKey(self.seed)
        X_data, y_data = self.generate_data(data_key)
        return (X_data, y_data), None


# TODO: Make separate MLP fixed-data class for experiments, when needed
@dataclass(frozen=True)
class SyntheticExperimentMLPTeacher(MLPStudentExperiment, ExperimentBase, SyntheticExperiment):
    """
    Defines parameters for a synthetic data experiment where the teacher is an MLP.
    This is a *fixed-time* experiment.

    MRO: SyntheticExperimentMLPTeacher -> MLPStudentExperiment -> ExperimentBase -> SyntheticExperiment -> object
    Model configuration comes from MLPStudentExperiment.
    Infrastructure (I/O, validation) comes from ExperimentBase.
    """

    TEACHER_INIT_KEY = 1

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

    def generate_teacher_weights(self):
        teacher_model = MLP(parameterization=self.teacher_parameterization, gamma=self.teacher_gamma)
        teacher_widths = [self.D] + [self.teacher_N] * (self.teacher_L - 1) + [1]
        # Use a fixed key for a deterministic teacher
        return teacher_model.init_params(init_key=self.TEACHER_INIT_KEY, widths=teacher_widths)

    def generate_data(self, data_key):
        """Generates a synthetic dataset using the MLP teacher."""
        teacher_weights = self.generate_teacher_weights()
        teacher_model = MLP(parameterization=self.teacher_parameterization, gamma=self.teacher_gamma)

        X_key, _ = jr.split(data_key, 2)
        X_data = jr.normal(X_key, (self.P, self.D))

        y_data = teacher_model(teacher_weights, X_data)

        return X_data, y_data

    def plot_title(self, task_name="MLP teacher", model_name="MLP"):
        line1 = (
            f"$T* = {self.num_steps}$ steps, {task_name} (Online) T(N={self.teacher_N}, L={self.teacher_L}), D={self.D}"
        )
        line2 = f"{model_name} in {self.parameterization.value} w/ $N={self.N}, L={self.L}, \\gamma={self.gamma}$"
        return f"{line1}\n{line2}"

    def should_skip_batch_size(self, batch_size: int, train_ds: any | None = None) -> bool:
        # Fixed-time experiments do not depend on dataset size.
        return False

    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        # For fixed-time experiments, num_epochs is not relevant. We can consider it as 1.
        return self.num_steps, 1

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        """Returns the runner for fixed-time synthetic experiments."""
        from .trainer import SyntheticFixedTimeTrialRunner

        return SyntheticFixedTimeTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        """Fixed-time experiments do not load a dataset upfront."""
        return None, None

    def is_online_experiment(self) -> bool:
        return True


@dataclass(frozen=True)
class SyntheticExperimentLinearTeacher(LinearStudentExperiment, ExperimentBase, SyntheticExperiment):
    r"""
    Defines parameters for a synthetic data experiment where the teacher is linear.
    This is a *fixed-data* experiment, with data generation process:

    -   Teacher: $y = w^{*T} x$
    -   Data features: $x \sim \mathcal{N}(0, \Sigma)$, where $\Sigma = \text{diag}(i^{-\alpha})_{i=1}^D$
    -   Teacher weights: $w_i^* = i^{-\theta}$, where $\theta = \frac{1}{2} + \frac{\alpha}{2}(\beta - 1)$

    Here $\alpha$ and $\beta$ are the capacity and source exponents as defined in
    https://abatanasov.com/Files/Scaling_Laws_Note.pdf, i.e., the decay rate of the
    data-covariance and of the tail of Var(y), respectively.
    """

    # Student and task parameters
    P: int

    # Teacher parameters
    alpha: float
    beta: float
    num_epochs: int = 1

    seed: int = 0  # Seed for reproducible data generation
    experiment_type: str = field(default="fixed_data_linear_teacher", init=False)

    def __post_init__(self):
        if self.D <= 0:
            raise ValueError(f"D must be positive, got {self.D}")
        if self.P <= 0:
            raise ValueError(f"P must be positive, got {self.P}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

    def generate_teacher_weights(self):
        r"""
        Generates the teacher weights $w^*$ according to the formula:
        $w_i^* = i^{-\theta}$, where $\theta = \frac{1}{2} + \frac{\alpha}{2}(\beta - 1)$.
        The weights are normalized such that $\text{Var}(y) = 1$ when $y = w^{*T}x$.
        """
        indices = np.arange(1, self.D + 1, dtype=np.float64)
        theta = 1 / 2 + 1 / 2 * self.alpha * (self.beta - 1)
        w = indices ** (-theta)

        # Normalize so that Var(y) = 1 for y = w^T x with x ~ N(0, Sigma)
        # Var(y) = sum_i w_i^2 * Sigma_ii, where Sigma_ii = i^{-alpha}
        variance = np.sum((indices ** (-self.alpha)) * (w**2))
        if variance > 0:
            w = w / np.sqrt(variance)

        return w.reshape(-1, 1)

    def generate_data(self, data_key):
        # x ~ N(0, Sigma), Sigma = diag(i^-alpha for i=1..D)
        # w_i = i^(-theta)
        # y = X @ w
        w = self.generate_teacher_weights()

        z_key, _ = jr.split(data_key, 2)
        z_data = jr.normal(z_key, (self.P, self.D))

        # Scale by sqrt(Sigma)
        # sqrt(Sigma) is a diagonal matrix with sqrt of Sigma's diagonal elements
        sigma_diag_sqrt = np.arange(1, self.D + 1) ** (-self.alpha / 2.0)
        X_data = z_data * sigma_diag_sqrt  # Element-wise multiplication, broadcasting

        y_data = X_data @ w
        return X_data, y_data

    def plot_title(self, task_name="linear task", model_name="Linear Model"):
        learning_type = "Online" if self.num_epochs == 1 else "Offline"
        line1 = (
            f"$P = {self.P}$ samples, {task_name} ({learning_type}) w/ "
            f"$D={self.D}, \\alpha={self.alpha}, \\beta={self.beta}$"
        )
        line2 = (
            "Student: "
            f"{model_name}, Epochs={self.num_epochs}, Optimizer: {self.optimizer.value}, "
            f"Loss: {self.loss_type.value}"
        )
        return f"{line1}\n{line2}"

    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        epochs_to_run = num_epochs if num_epochs is not None else self.num_epochs
        steps_per_epoch = self.P // batch_size
        return epochs_to_run * steps_per_epoch, epochs_to_run

    def should_skip_batch_size(self, batch_size: int, train_ds: any | None = None) -> bool:
        # P is the authoritative dataset size for this experiment type.
        if batch_size <= 0 or batch_size > self.P:
            logging.warning(f"Skipping batch size {batch_size} > dataset size P ({self.P}).")
            return True
        return False

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        """Returns the runner for fixed-data synthetic experiments."""
        from .trainer import SyntheticFixedDataTrialRunner

        return SyntheticFixedDataTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        """Generates the synthetic dataset for this experiment."""
        data_key = jr.PRNGKey(self.seed)
        X_data, y_data = self.generate_data(data_key)
        return (X_data, y_data), None

    def get_test_error_fn(self, init_key: int | None = None):
        """
        Returns a callable mapping model parameters -> test error for this teacher.

        If `init_key` is None, this returns the population MSE (½ ∥·∥² + noise term).
        If `init_key` is provided, the returned function evaluates the empirical
        MSE on the deterministic evaluation dataset used during training (matching
        the `final_eval_loss` metric stored in results).
        """
        rho = getattr(self, "rho", 0.0)

        if init_key is None:
            teacher_vec = jnp.asarray(self.generate_teacher_weights()).reshape(-1)
            sigma_diag = jnp.arange(1, self.D + 1, dtype=teacher_vec.dtype) ** (-self.alpha)

            def population_error(params):
                flat_params, _ = ravel_pytree(params)
                diff = flat_params - teacher_vec
                mse = jnp.sum(diff**2 * sigma_diag)
                return 0.5 * ((1.0 - rho) * mse + rho)

            return population_error

        eval_key = jr.PRNGKey(init_key + SYNTH_EVAL_DATA_SEED_OFFSET)
        X_eval, y_eval = self.generate_data(eval_key)
        X_eval = jnp.asarray(X_eval)
        y_eval = jnp.asarray(y_eval)
        max_samples = SYNTH_EVAL_MAX_SAMPLES
        if X_eval.shape[0] > max_samples:
            subset_key = jr.PRNGKey(init_key + SYNTH_EVAL_SUBSET_SEED_OFFSET)
            indices = jr.permutation(subset_key, X_eval.shape[0])[:max_samples]
            X_eval = X_eval[indices]
            y_eval = y_eval[indices]

        def empirical_error(params):
            flat_params, _ = ravel_pytree(params)
            weights = flat_params.reshape(self.D, -1)
            preds = X_eval @ weights
            diff = y_eval - preds
            return 0.5 * jnp.mean(diff**2)

        return empirical_error


@dataclass(frozen=True)
class SyntheticExperimentNoisyLinearTeacher(SyntheticExperimentLinearTeacher):
    r"""
    Linear teacher with additive Gaussian noise:
    $y = \sqrt{1 - \rho} \cdot x^T w^* + \sqrt{\rho} \cdot \varepsilon$, where $\varepsilon \sim \mathcal{N}(0, I)$.
    The clean signal $x^T w^*$ is normalized to unit variance; $\rho \in [0, 1]$ controls noise strength.
    """

    rho: float = 0.0
    experiment_type: str = field(default="fixed_data_noisy_linear_teacher", init=False)

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {self.rho}")

    def signal_to_noise(self) -> float:
        if self.rho == 0.0:
            return np.inf
        return (1.0 - self.rho) / self.rho

    def generate_data(self, data_key):
        rho = float(self.rho)
        X_data, y_clean = super().generate_data(data_key)
        if rho == 0.0:
            return X_data, y_clean

        clean_scale = np.sqrt(1.0 - rho)
        noise_scale = np.sqrt(rho)
        noise_key = jr.fold_in(data_key, 1)

        noise = jr.normal(noise_key, y_clean.shape)
        y_noisy = clean_scale * y_clean + noise_scale * noise
        return X_data, y_noisy

    def plot_title(self, task_name="linear task", model_name="Linear Model"):
        base = super().plot_title(task_name=task_name, model_name=model_name)
        snr = self.signal_to_noise()
        if np.isinf(snr):
            noise_line = "Noise: ρ = 0 (clean)"
        else:
            noise_line = f"Noise: ρ = {self.rho:.2f}, SNR={(snr):.2f}"
        return f"{base}\n{noise_line}"


@dataclass(frozen=True)
class MNISTExperiment(MLPStudentExperiment, ExperimentBase):
    """
    Defines parameters for a 10-class MNIST classification experiment.
    This is a fixed-data, fixed-epoch experiment that uses the custom MLP
    from models.py. Sweeps are performed over batch size and learning rate.

    MRO: MNISTExperiment -> MLPStudentExperiment -> ExperimentBase -> object
    Model configuration comes from MLPStudentExperiment.
    Infrastructure (I/O, validation) comes from ExperimentBase.
    """

    D: int = field(default=784, init=False)  # Input dim for flattened MNIST
    num_outputs: int = 10

    # Training parameters
    num_epochs: int = 1

    experiment_type: str = field(default="mnist_classification", init=False)

    def __post_init__(self):
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

    def get_output_dim(self) -> int:
        return self.num_outputs

    def plot_title(self, task_name="MNIST Classification", model_name="MLP"):
        learning_type = "Online" if self.num_epochs == 1 else "Offline"
        line1 = (
            f"{task_name} ({learning_type}) ("
            f"{model_name} N={self.N}, L={self.L}, {self.parameterization.value}, $\\gamma={self.gamma}$)"
        )
        line2 = f"Epochs={self.num_epochs}, Optimizer={self.optimizer.value}"
        return f"{line1}\n{line2}"

    def should_skip_batch_size(self, batch_size: int, train_ds: any | None = None) -> bool:
        if train_ds is None:
            # In a pre-flight check, we don't know the dataset size, so we can't skip.
            return False
        train_ds_size = len(train_ds["image"])
        if batch_size > train_ds_size:
            logging.warning(f"Skipping batch size {batch_size} > dataset size ({train_ds_size}).")
            return True
        return False

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        """Returns the runner for MNIST experiments."""
        from .trainer import MNISTTrialRunner

        return MNISTTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        """Loads the MNIST dataset."""
        return _load_mnist_dataset(
            self,
            init_key,
            dataset_loader=kwargs.get("dataset_loader"),
            forced_subsample_seed=kwargs.get("forced_subsample_seed"),
        )

    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        assert train_ds is not None, "compute_num_steps requires a non-None train_ds for this experiment."
        epochs_to_run = num_epochs if num_epochs is not None else self.num_epochs
        num_train_samples = len(train_ds["image"])
        steps_per_epoch = num_train_samples // batch_size
        return epochs_to_run * steps_per_epoch, epochs_to_run

    def is_classification(self) -> bool:
        return True


@dataclass(frozen=True)
class MNIST1MExperiment(MLPStudentExperiment, ExperimentBase):
    """
    Defines parameters for a 10-class MNIST-1M classification experiment.
    This dataset is generated from a diffusion model.

    MRO: MNIST1MExperiment -> MLPStudentExperiment -> ExperimentBase -> object
    Model configuration comes from MLPStudentExperiment.
    Infrastructure (I/O, validation) comes from ExperimentBase.
    """

    num_epochs: int
    D: int = field(default=784, init=False)
    num_outputs: int = 10

    experiment_type: str = field(default="mnist1m_classification", init=False)

    def __post_init__(self):
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

    def get_output_dim(self) -> int:
        return self.num_outputs

    def plot_title(self, task_name="MNIST-1M Classification", model_name="MLP"):
        learning_type = "Online" if self.num_epochs == 1 else "Offline"
        line1 = (
            f"{task_name} ({learning_type}) ("
            f"{model_name} N={self.N}, L={self.L}, {self.parameterization.value}, $\\gamma={self.gamma}$)"
        )
        line2 = f"Epochs={self.num_epochs}, Optimizer={self.optimizer.value}, " f"Loss={self.loss_type.value}"
        return f"{line1}\n{line2}"

    def should_skip_batch_size(self, batch_size: int, train_ds: any | None = None) -> bool:
        if train_ds is None:
            # In a pre-flight check, we don't know the dataset size, so we can't skip.
            return False
        train_ds_size = len(train_ds["image"])
        if batch_size > train_ds_size:
            logging.warning(f"Skipping batch size {batch_size} > dataset size ({train_ds_size}).")
            return True
        return False

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        """Returns the runner for MNIST experiments."""
        from .trainer import MNISTTrialRunner

        return MNISTTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        """Loads the MNIST-1M dataset."""
        return _load_mnist_dataset(
            self,
            init_key,
            dataset_loader=kwargs.get("dataset_loader"),
            forced_subsample_seed=kwargs.get("forced_subsample_seed"),
        )

    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        assert train_ds is not None, "compute_num_steps requires a non-None train_ds for this experiment."
        epochs_to_run = num_epochs if num_epochs is not None else self.num_epochs
        num_train_samples = len(train_ds["image"])
        steps_per_epoch = num_train_samples // batch_size
        return epochs_to_run * steps_per_epoch, epochs_to_run

    def is_classification(self) -> bool:
        return True


@dataclass(frozen=True)
class MNIST1MSampledExperiment(MLPStudentExperiment, ExperimentBase):
    """
    An MNIST-1M experiment that trains on a subset of the full dataset.
    This is useful for quick sanity checks and faster experimental cycles.

    MRO: MNIST1MSampledExperiment -> MLPStudentExperiment -> ExperimentBase -> object
    Model configuration comes from MLPStudentExperiment.
    Infrastructure (I/O, validation) comes from ExperimentBase.
    """

    num_epochs: int
    max_train_samples: int
    num_outputs: int = 10
    D: int = field(default=784, init=False)
    experiment_type: str = field(default="mnist1m_sampled_classification", init=False)

    def __post_init__(self):
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.max_train_samples <= 0:
            raise ValueError(f"max_train_samples must be positive, got {self.max_train_samples}")

    def get_output_dim(self) -> int:
        return self.num_outputs

    def plot_title(self, task_name="MNIST-1M Sampled", model_name="MLP"):
        learning_type = "Online" if self.num_epochs == 1 else "Offline"
        line1 = (
            f"{task_name} ({learning_type}) ("
            f"{model_name} N={self.N}, L={self.L}, {self.parameterization.value}, $\\gamma={self.gamma}$)"
        )
        line2 = (
            f"Epochs={self.num_epochs}, Optimizer={self.optimizer.value}, "
            f"Loss={self.loss_type.value}, Samples={self.max_train_samples}"
        )
        return f"{line1}\n{line2}"

    def should_skip_batch_size(self, batch_size: int, train_ds: any | None = None) -> bool:
        # For sampled experiments, the config's `max_train_samples` is the
        # effective dataset size we should check against.
        effective_size = self.max_train_samples
        if train_ds is not None:
            effective_size = min(self.max_train_samples, len(train_ds["image"]))
        if batch_size > effective_size:
            logging.warning(f"Skipping batch size {batch_size} > effective dataset size ({effective_size}).")
            return True
        return False

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        """Returns the runner for MNIST experiments."""
        from .trainer import MNISTTrialRunner

        return MNISTTrialRunner

    def get_sweep_metadata(self, init_key: int) -> dict:
        """This experiment uses a random subset of data, so we save the seed."""
        return {"subsample_seed": init_key}

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        """Loads and subsamples the MNIST-1M dataset."""
        return _load_mnist_dataset(
            self,
            init_key,
            dataset_loader=kwargs.get("dataset_loader"),
            forced_subsample_seed=kwargs.get("forced_subsample_seed"),
        )

    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        assert train_ds is not None, "compute_num_steps requires a non-None train_ds for this experiment."
        epochs_to_run = num_epochs if num_epochs is not None else self.num_epochs
        num_train_samples = len(train_ds["image"])
        steps_per_epoch = num_train_samples // batch_size
        return epochs_to_run * steps_per_epoch, epochs_to_run

    def is_classification(self) -> bool:
        return True
