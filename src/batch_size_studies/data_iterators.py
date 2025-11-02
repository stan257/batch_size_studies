from abc import ABC, abstractmethod
from typing import Generator

import jax.random as jr
import numpy as np


class DataIterator(ABC):
    """Abstract base class for data iterators."""

    @abstractmethod
    def __iter__(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yields batches of (inputs, targets)."""
        raise NotImplementedError


class EpochBasedDataIterator(DataIterator):
    """
    Iterates over a fixed dataset for a specified number of epochs.

    Handles shuffling data at the start of each epoch and can resume
    from a specific step.
    """

    def __init__(self, train_ds, batch_size, num_epochs, init_key, start_step=0):
        self.train_ds = train_ds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.init_key = init_key
        self.start_step = start_step

        if isinstance(train_ds, dict):  # MNIST-like
            self.original_num_train = self.train_ds["image"].shape[0]
        else:  # Synthetic-like (X, y)
            self.original_num_train = self.train_ds[0].shape[0]

        self.steps_per_epoch = self.original_num_train // self.batch_size
        num_usable_samples = self.steps_per_epoch * self.batch_size

        # Create a fixed subset of indices for the entire trial
        subset_key = jr.PRNGKey(self.init_key)
        self.subset_indices = jr.permutation(subset_key, self.original_num_train)[:num_usable_samples]

    def __iter__(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        start_epoch = self.start_step // self.steps_per_epoch
        step_in_epoch = self.start_step % self.steps_per_epoch

        for epoch in range(start_epoch, self.num_epochs):
            epoch_shuffle_key = jr.PRNGKey(self.init_key + epoch + 1)
            perms = jr.permutation(epoch_shuffle_key, self.subset_indices)
            epoch_perms = perms.reshape((self.steps_per_epoch, self.batch_size))

            for i, perm in enumerate(np.array(epoch_perms)):
                if i >= step_in_epoch:
                    if isinstance(self.train_ds, dict):  # MNIST-like
                        batch_images = self.train_ds["image"][perm, ...].reshape(self.batch_size, -1)
                        batch_labels = self.train_ds["label"][perm, ...]
                        yield batch_images, batch_labels
                    else:  # Synthetic-like
                        yield self.train_ds[0][perm, ...], self.train_ds[1][perm, ...]
            step_in_epoch = 0  # Reset for subsequent epochs


class OnlineDataIterator(DataIterator):
    """Generates new data on-the-fly for online training paradigms."""

    def __init__(self, experiment, batch_size, results, start_step=0):
        self.experiment = experiment
        self.batch_size = batch_size
        self.start_step = start_step
        self.results = results

    def __iter__(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        batch_key_seed = self.results.get("batch_key_seed", 0)
        steps_per_batch_key = self.experiment.P // self.batch_size
        batch_key_seed += self.start_step // steps_per_batch_key
        step_for_curr_data = self.start_step % steps_per_batch_key

        while True:  # The training loop in TrialRunner will stop this
            if (step_for_curr_data + 1) * self.batch_size > self.experiment.P:
                batch_key_seed += 1
                step_for_curr_data = 0

            X_data, y_data = self.experiment.generate_data(jr.key(batch_key_seed))
            self.results["batch_key_seed"] = batch_key_seed

            start = step_for_curr_data * self.batch_size
            X_batch, y_batch = (
                X_data[start : start + self.batch_size],
                y_data[start : start + self.batch_size],
            )
            step_for_curr_data += 1
            yield X_batch, y_batch
