from typing import Generator

import jax.random as jr
import numpy as np

from .constants import EPOCH_SHUFFLE_SEED_OFFSET
from .protocols import DataIterator


class EpochBasedDataIterator(DataIterator):
    """
    Iterates over a fixed dataset for a specified number of epochs.

    Handles shuffling data at the start of each epoch and can resume
    from a specific step.
    """

    def __init__(self, train_ds, batch_size, num_epochs, init_key, start_step=0, resume_state: dict | None = None):
        self.train_ds = train_ds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.init_key = init_key
        self.start_step = start_step
        self.resume_state = resume_state or {}

        if isinstance(train_ds, dict):  # MNIST-like
            self.original_num_train = self.train_ds["image"].shape[0]
        else:  # Synthetic-like (X, y)
            self.original_num_train = self.train_ds[0].shape[0]

        self.steps_per_epoch = self.original_num_train // self.batch_size
        num_usable_samples = self.steps_per_epoch * self.batch_size
        # Drop the remainder so every run sees the same number of steps; leftover samples are never used.

        # Create a fixed subset of indices for the entire trial
        subset_key = jr.PRNGKey(self.init_key)
        self.subset_indices = jr.permutation(subset_key, self.original_num_train)[:num_usable_samples]

    def __iter__(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        if self.steps_per_epoch == 0:
            return

        start_epoch = self.start_step // self.steps_per_epoch
        step_in_epoch = self.start_step % self.steps_per_epoch
        stored_step_in_epoch = self.resume_state.get("step_in_epoch")
        if stored_step_in_epoch is not None:
            step_in_epoch = int(stored_step_in_epoch)

        resume_epoch = self.resume_state.get("epoch")
        resume_seed = self.resume_state.get("epoch_seed")
        for epoch in range(start_epoch, self.num_epochs):
            if resume_seed is not None and resume_epoch == epoch:
                epoch_shuffle_key = jr.PRNGKey(int(resume_seed))
                resume_seed = None  # Use override only once
            else:
                epoch_shuffle_key = jr.PRNGKey(self.init_key + epoch + EPOCH_SHUFFLE_SEED_OFFSET)
            perms = jr.permutation(epoch_shuffle_key, self.subset_indices)
            epoch_perms = perms.reshape((self.steps_per_epoch, self.batch_size))

            for i, perm in enumerate(np.array(epoch_perms)):
                if i >= step_in_epoch:
                    if isinstance(self.train_ds, dict):  # MNIST-like
                        batch_images = self.train_ds["image"][perm, ...]
                        if batch_images.ndim > 2:
                            batch_images = batch_images.reshape(self.batch_size, -1)
                        batch_labels = self.train_ds["label"][perm, ...]
                        yield batch_images, batch_labels
                    else:  # Synthetic-like
                        yield self.train_ds[0][perm, ...], self.train_ds[1][perm, ...]
            step_in_epoch = 0  # Reset for subsequent epochs


class OnlineDataIterator(DataIterator):
    """Generates new data on-the-fly for online training paradigms."""

    def __init__(self, experiment, batch_size, start_step=0, initial_batch_key_seed=0):
        self.experiment = experiment
        self.batch_size = batch_size
        self.start_step = start_step
        self.initial_batch_key_seed = initial_batch_key_seed
        self.current_batch_key_seed = initial_batch_key_seed

    def __iter__(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        if self.batch_size <= 0:
            return

        steps_per_batch_key = self.experiment.P // self.batch_size
        if steps_per_batch_key == 0:
            # This handles the case where batch_size > P
            # The iterator will correctly yield no batches.
            return

        batch_key_seed = self.initial_batch_key_seed
        batch_key_seed += self.start_step // steps_per_batch_key
        step_in_block = self.start_step % steps_per_batch_key

        # Initial data generation before the loop starts
        X_data, y_data = self.experiment.generate_data(jr.PRNGKey(batch_key_seed))
        self.current_batch_key_seed = batch_key_seed

        while True:  # The training loop in TrialRunner will stop this
            # Check if we need to generate a new block of data
            if (step_in_block + 1) * self.batch_size > self.experiment.P:
                batch_key_seed += 1
                step_in_block = 0
                X_data, y_data = self.experiment.generate_data(jr.PRNGKey(batch_key_seed))
                self.current_batch_key_seed = batch_key_seed

            # Yield a batch from the current data block
            start = step_in_block * self.batch_size
            X_batch, y_batch = (
                X_data[start : start + self.batch_size],
                y_data[start : start + self.batch_size],
            )
            step_in_block += 1
            yield X_batch, y_batch
