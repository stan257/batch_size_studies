from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Type

import jax.random as jr
import numpy as np

from ..data_loading import load_datasets, load_mnist1m_dataset
from ..protocols import TrialRunner
from .base import ExperimentBase, MLPStudentExperiment


def _subsample_mnist_data(train_images, train_labels, experiment, init_key):
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
    if dataset_loader is None:
        dataset_loader = load_datasets if isinstance(experiment, MNISTExperiment) else load_mnist1m_dataset
    try:
        (train_images, train_labels), (test_images, test_labels) = dataset_loader()
        if isinstance(experiment, MNIST1MSampledExperiment):
            seed_to_use = forced_subsample_seed if forced_subsample_seed is not None else init_key
            train_images, train_labels = _subsample_mnist_data(train_images, train_labels, experiment, seed_to_use)
        train_ds = {"image": train_images, "label": train_labels}
        test_ds = {"image": test_images, "label": test_labels}
        return train_ds, test_ds
    except Exception as e:
        logging.error(f"Failed to load dataset: {type(e).__name__}: {e}")
        return None, None


@dataclass(frozen=True)
class MNISTExperiment(MLPStudentExperiment, ExperimentBase):
    D: int = field(default=784, init=False)
    num_outputs: int = 10
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
            return False
        train_ds_size = len(train_ds["image"])
        if batch_size > train_ds_size:
            logging.warning(f"Skipping batch size {batch_size} > dataset size ({train_ds_size}).")
            return True
        return False

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        from ..trainer import MNISTTrialRunner

        return MNISTTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        return _load_mnist_dataset(
            self,
            init_key,
            dataset_loader=kwargs.get("dataset_loader"),
            forced_subsample_seed=kwargs.get("forced_subsample_seed"),
        )

    def get_default_dataset_loader(self):
        return load_datasets

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
        line2 = f"Epochs={self.num_epochs}, Optimizer={self.optimizer.value}, Loss={self.loss_type.value}"
        return f"{line1}\n{line2}"

    def should_skip_batch_size(self, batch_size: int, train_ds: any | None = None) -> bool:
        if train_ds is None:
            return False
        train_ds_size = len(train_ds["image"])
        if batch_size > train_ds_size:
            logging.warning(f"Skipping batch size {batch_size} > dataset size ({train_ds_size}).")
            return True
        return False

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        from ..trainer import MNISTTrialRunner

        return MNISTTrialRunner

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        return _load_mnist_dataset(
            self,
            init_key,
            dataset_loader=kwargs.get("dataset_loader"),
            forced_subsample_seed=kwargs.get("forced_subsample_seed"),
        )

    def get_default_dataset_loader(self):
        return load_mnist1m_dataset

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
        effective_size = self.max_train_samples
        if train_ds is not None:
            effective_size = min(self.max_train_samples, len(train_ds["image"]))
        if batch_size > effective_size:
            logging.warning(f"Skipping batch size {batch_size} > effective dataset size ({effective_size}).")
            return True
        return False

    def get_trial_runner_class(self) -> Type[TrialRunner]:
        from ..trainer import MNISTTrialRunner

        return MNISTTrialRunner

    def get_sweep_metadata(self, init_key: int) -> dict:
        return {"subsample_seed": init_key}

    def prepare_datasets(self, init_key: int, **kwargs) -> tuple[any, any]:
        return _load_mnist_dataset(
            self,
            init_key,
            dataset_loader=kwargs.get("dataset_loader"),
            forced_subsample_seed=kwargs.get("forced_subsample_seed"),
        )

    def get_default_dataset_loader(self):
        return load_mnist1m_dataset

    def compute_num_steps(self, batch_size: int, train_ds: any, num_epochs: int | None) -> tuple[int, int]:
        assert train_ds is not None, "compute_num_steps requires a non-None train_ds for this experiment."
        epochs_to_run = num_epochs if num_epochs is not None else self.num_epochs
        num_train_samples = len(train_ds["image"])
        steps_per_epoch = num_train_samples // batch_size
        return epochs_to_run * steps_per_epoch, epochs_to_run

    def is_classification(self) -> bool:
        return True
