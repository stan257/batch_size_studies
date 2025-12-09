from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.flatten_util import ravel_pytree

from ..constants import SYNTH_EVAL_DATA_SEED_OFFSET, SYNTH_EVAL_MAX_SAMPLES, SYNTH_EVAL_SUBSET_SEED_OFFSET
from ..definitions import Parameterization
from ..models import MLP
from ..protocols import TrialRunner
from .base import ExperimentBase, LinearStudentExperiment, MLPStudentExperiment


class SyntheticExperiment(ABC):
    @abstractmethod
    def generate_teacher_weights(self): ...

    @abstractmethod
    def generate_data(self, data_key): ...


@dataclass(frozen=True)
class SyntheticExperimentFixedTime(MLPStudentExperiment, ExperimentBase, SyntheticExperiment):
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
        if batch_size <= 0 or batch_size > self.P:
            logging.warning(f"Skipping batch size {batch_size} > synthetic block size P ({self.P}).")
            return True
        return False

    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        return self.num_steps, 1

    def is_online_experiment(self) -> bool:
        return True

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        from ..trainer import SyntheticFixedTimeTrialRunner

        return SyntheticFixedTimeTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        return None, None


@dataclass(frozen=True)
class SyntheticExperimentFixedData(MLPStudentExperiment, ExperimentBase, SyntheticExperiment):
    D: int
    P: int
    K: int
    num_epochs: int = 1
    seed: int = 0
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
        if batch_size <= 0 or batch_size > self.P:
            logging.warning(f"Skipping batch size {batch_size} > dataset size P ({self.P}).")
            return True
        return False

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        from ..trainer import SyntheticFixedDataTrialRunner

        return SyntheticFixedDataTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        data_key = jr.PRNGKey(self.seed)
        X_data, y_data = self.generate_data(data_key)
        return (X_data, y_data), None


@dataclass(frozen=True)
class SyntheticExperimentMLPTeacher(MLPStudentExperiment, ExperimentBase, SyntheticExperiment):
    TEACHER_INIT_KEY = 1

    D: int
    P: int
    num_steps: int

    teacher_N: int
    teacher_L: int
    teacher_gamma: float
    teacher_parameterization: Parameterization

    experiment_type: str = field(default="fixed_time_mlp_teacher", init=False)

    def generate_teacher_weights(self):
        teacher_model = MLP(parameterization=self.teacher_parameterization, gamma=self.teacher_gamma)
        teacher_widths = [self.D] + [self.teacher_N] * (self.teacher_L - 1) + [1]
        return teacher_model.init_params(init_key=self.TEACHER_INIT_KEY, widths=teacher_widths)

    def generate_data(self, data_key):
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
        return False

    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        return self.num_steps, 1

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        from ..trainer import SyntheticFixedTimeTrialRunner

        return SyntheticFixedTimeTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        return None, None

    def is_online_experiment(self) -> bool:
        return True


@dataclass(frozen=True)
class SyntheticExperimentLinearTeacher(LinearStudentExperiment, ExperimentBase, SyntheticExperiment):
    P: int
    alpha: float
    beta: float
    num_epochs: int = 1
    seed: int = 0
    experiment_type: str = field(default="fixed_data_linear_teacher", init=False)

    def __post_init__(self):
        if self.D <= 0:
            raise ValueError(f"D must be positive, got {self.D}")
        if self.P <= 0:
            raise ValueError(f"P must be positive, got {self.P}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

    def generate_teacher_weights(self):
        indices = np.arange(1, self.D + 1, dtype=np.float64)
        theta = 1 / 2 + 1 / 2 * self.alpha * (self.beta - 1)
        w = indices ** (-theta)
        variance = np.sum((indices ** (-self.alpha)) * (w**2))
        if variance > 0:
            w = w / np.sqrt(variance)
        return w.reshape(-1, 1)

    def generate_data(self, data_key):
        w = self.generate_teacher_weights()
        z_key, _ = jr.split(data_key, 2)
        z_data = jr.normal(z_key, (self.P, self.D))
        sigma_diag_sqrt = np.arange(1, self.D + 1) ** (-self.alpha / 2.0)
        X_data = z_data * sigma_diag_sqrt
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
        if batch_size <= 0 or batch_size > self.P:
            logging.warning(f"Skipping batch size {batch_size} > dataset size P ({self.P}).")
            return True
        return False

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        from ..trainer import SyntheticFixedDataTrialRunner

        return SyntheticFixedDataTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        data_key = jr.PRNGKey(self.seed)
        X_data, y_data = self.generate_data(data_key)
        return (X_data, y_data), None

    def get_test_error_fn(self, init_key: int | None = None):
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
