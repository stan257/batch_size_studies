import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data_iterators import DataIterator, EpochBasedDataIterator, OnlineDataIterator
from .definitions import LossType
from .protocols import DivergenceError, TrialRunner
from .training_utils import create_base_optimizer_transform

if TYPE_CHECKING:
    from .runner import TrialContext


class EpochBasedTrialRunner(TrialRunner):
    """Shared helper for fixed-data runners that iterate by epochs."""

    def __init__(self, context):
        super().__init__(context)
        self.num_epochs = context.num_epochs
        self.train_ds = context.train_ds
        self.init_key = context.init_key
        self.steps_per_epoch = self._compute_steps_per_epoch()
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive for epoch-based runners.")

    def _compute_steps_per_epoch(self) -> int:
        train_size = self._get_dataset_size(self.train_ds)
        return train_size // self.run_key.batch_size

    def _get_dataset_size(self, train_ds) -> int:
        if isinstance(train_ds, dict):
            return train_ds["image"].shape[0]
        return train_ds[0].shape[0]

    def _get_iterator_init_key(self) -> int:
        """Allow subclasses to override how the iterator seed is chosen."""
        return self.init_key

    def _create_data_iterator(self, start_step: int, results: dict) -> EpochBasedDataIterator:
        return EpochBasedDataIterator(
            train_ds=self.train_ds,
            batch_size=self.run_key.batch_size,
            num_epochs=self.num_epochs,
            init_key=self._get_iterator_init_key(),
            start_step=start_step,
            resume_state=results.get("iterator_state"),
        )

    def _post_step_hook(self, step: int, params, results: dict, aux: Any) -> dict:
        next_step = step + 1
        if self.steps_per_epoch > 0:
            epoch_index = next_step // self.steps_per_epoch
            results["iterator_state"] = {
                "global_step": next_step,
                "epoch": epoch_index,
                "step_in_epoch": next_step % self.steps_per_epoch,
                "epoch_seed": self._get_epoch_seed(epoch_index),
            }
            if next_step % self.steps_per_epoch == 0:
                epoch = self._step_to_completed_epoch(step)
                return self._on_epoch_end(epoch, params, results, aux)
        return results

    def _step_to_completed_epoch(self, step: int) -> int:
        """Converts a training step (0-indexed) to a completed epoch number (0-indexed)."""
        return (step + 1) // self.steps_per_epoch - 1

    def _post_training_hook(self, params, results: dict) -> dict:
        results.pop("iterator_state", None)
        return super()._post_training_hook(params, results)

    def _adjust_start_step(self, start_step: int, results: dict) -> int:
        resume_state = results.get("iterator_state")
        if isinstance(resume_state, dict):
            return int(resume_state.get("global_step", start_step))
        return start_step

    def _get_epoch_seed(self, epoch_index: int) -> int:
        # Use a deterministic, unique seed for each epoch for reproducibility.
        return int(self._get_iterator_init_key() + epoch_index + 1)

    @abstractmethod
    def _on_epoch_end(self, epoch: int, params, results: dict, aux: Any) -> dict:
        """Hook invoked whenever an epoch boundary is reached."""
        raise NotImplementedError


class MNISTTrialRunner(EpochBasedTrialRunner):
    """Trial runner for MNIST-based experiments."""

    EVAL_BATCH_SIZE = 512

    def __init__(self, context):
        super().__init__(context)
        self.test_ds = context.test_ds
        self.max_eval_samples = context.kwargs.get("max_eval_samples", 16_384)  # = 2^14
        save_dense_snapshots = context.kwargs.get("save_interstitial_snapshots", False)
        self.snapshot_steps = self._compute_snapshot_steps(self.num_steps, save_dense_snapshots)
        self._ensure_epoch_snapshot_steps(self.steps_per_epoch, self.num_epochs)
        if self.snapshot_steps and self.snapshot_steps[0] == 0:
            self.snapshot_steps = [step for step in self.snapshot_steps if step != 0]

    def _init_results(self) -> dict:
        return {
            "epoch_test_accuracies": [],
            "loss_history": [],
            "expected_epochs": self.num_epochs,
        }

    def _create_loss_fn(self) -> Callable:
        apply_fn = self.model_instance
        match self.experiment.loss_type:
            case LossType.XENT:

                def loss_fn(params, x_batch, y_batch_labels):
                    logits = apply_fn(params, x_batch)
                    loss = jnp.mean(
                        optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y_batch_labels)
                    )
                    return loss, logits

                return loss_fn
            case LossType.MSE:

                def loss_fn(params, x_batch, y_batch_labels):
                    logits = apply_fn(params, x_batch)
                    one_hot_labels = jax.nn.one_hot(y_batch_labels, num_classes=self.experiment.num_outputs)
                    diff = logits - one_hot_labels
                    diff = diff.reshape(diff.shape[0], -1)
                    loss = 0.5 * jnp.mean(jnp.sum(diff**2, axis=1))
                    return loss, logits

                return loss_fn
            case _:
                raise NotImplementedError(f"Loss type {self.experiment.loss_type} not implemented.")

    def _create_jitted_update_step(
        self, loss_fn: Callable, base_optimizer_transform: optax.GradientTransformation
    ) -> Callable:
        """Creates and JIT-compiles the update step function for MNIST."""

        def update_step_fn(params, opt_state, x_batch, y_batch, lr):
            optimizer = optax.chain(base_optimizer_transform, optax.scale(-lr))
            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_batch, y_batch)
            # Adam needs params in update, SGD does not. Pass it to be safe.
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            accuracy = jnp.mean(jnp.argmax(logits, -1) == y_batch)
            return new_params, new_opt_state, loss, accuracy

        return jax.jit(update_step_fn)

    def _create_eval_step(self) -> Callable:
        apply_fn = self.model_instance

        @jax.jit
        def eval_step(params, x_batch, y_batch):
            logits = apply_fn(params, x_batch)
            accuracy = jnp.mean(jnp.argmax(logits, -1) == y_batch)
            return accuracy

        return eval_step

    def _create_data_iterator(self, start_step: int, results: dict) -> EpochBasedDataIterator:
        return EpochBasedDataIterator(
            train_ds=self.train_ds,
            batch_size=self.run_key.batch_size,
            num_epochs=self.num_epochs,
            init_key=self.init_key,
            start_step=start_step,
            resume_state=results.get("iterator_state"),
        )

    def _on_epoch_end(self, epoch: int, params, results: dict, aux: Any) -> dict:
        if self.test_ds is None:
            return results
        if self.pbar:
            self.pbar.set_description(
                f"Sweep (B={self.run_key.batch_size}, eta={self.run_key.eta:.3g}) | Epoch {epoch + 2}/{self.num_epochs}"
            )
        test_accuracies = []
        images = self.test_ds["image"]
        labels = self.test_ds["label"]
        num_test_samples = images.shape[0]
        eval_samples = min(num_test_samples, self.max_eval_samples or num_test_samples)
        if eval_samples < num_test_samples:
            eval_key = jax.random.PRNGKey(self.init_key + epoch + 17)
            indices = np.array(jax.random.permutation(eval_key, num_test_samples)[:eval_samples])
            images = images[indices]
            labels = labels[indices]
            num_test_samples = eval_samples
        if images.ndim > 2:
            images = images.reshape(num_test_samples, -1)
        for i in range((num_test_samples + self.EVAL_BATCH_SIZE - 1) // self.EVAL_BATCH_SIZE):
            start_idx = i * self.EVAL_BATCH_SIZE
            end_idx = (i + 1) * self.EVAL_BATCH_SIZE
            batch_images = images[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            test_accuracies.append(self.eval_step(params, batch_images, batch_labels))

        epoch_accuracy = float(jnp.mean(jnp.array(test_accuracies)))
        results["epoch_test_accuracies"].append(epoch_accuracy)
        if self.pbar:
            self.pbar.set_postfix(accuracy=f"{epoch_accuracy:.4f}")

        return results

    def _should_save_checkpoint(self, step: int) -> bool:
        return step in self.snapshot_steps

    def _post_training_hook(self, params, results: dict) -> dict:
        """Sets the final accuracy metric after the training loop completes."""
        results = super()._post_training_hook(params, results)
        if results.get("epoch_test_accuracies"):
            results["final_test_accuracy"] = results["epoch_test_accuracies"][-1]
        return results

    def is_complete(self, result: dict) -> bool:
        """A run is complete if the number of test accuracies matches the expected number of epochs."""
        return len(result.get("epoch_test_accuracies", [])) >= result.get("expected_epochs", self.num_epochs)


class SyntheticTrialRunner(TrialRunner):
    """Base trial runner for synthetic data experiments."""

    EVAL_MAX_SAMPLES = 10_000

    def __init__(self, context):
        super().__init__(context)
        save_dense_snapshots = context.kwargs.get("save_interstitial_snapshots", False)
        self.snapshot_steps = self._compute_snapshot_steps(context.num_steps, save_dense_snapshots)
        disable_eval_ds = context.kwargs.get("disable_eval_dataset", False)
        self.eval_ds = None if disable_eval_ds else self._create_eval_dataset(context.init_key)

    def _create_loss_fn(self) -> Callable:
        def loss_fn(params, x_batch, y_batch):
            pred = self.model_instance(params, x_batch)
            diff = y_batch - pred
            diff = diff.reshape(diff.shape[0], -1)
            loss = 0.5 * jnp.mean(jnp.sum(diff**2, axis=1))
            # Return a dummy aux output for a consistent interface with MNISTTrialRunner
            return loss, None

        return loss_fn

    def _create_jitted_update_step(
        self, loss_fn: Callable, base_optimizer_transform: optax.GradientTransformation
    ) -> Callable:
        """Creates and JIT-compiles the update step function for synthetic experiments."""

        def update_step_fn(params, opt_state, x_batch, y_batch, lr):
            optimizer = optax.chain(base_optimizer_transform, optax.scale(-lr))
            (loss, _), grad = jax.value_and_grad(loss_fn, has_aux=True)(params, x_batch, y_batch)
            updates, new_opt_state = optimizer.update(grad, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            # Return a dummy aux output for a consistent signature
            return new_params, new_opt_state, loss, None

        return jax.jit(update_step_fn)

    def _create_eval_dataset(self, init_key: int):
        """Generates a deterministic evaluation dataset for synthetic experiments, if supported."""
        if not hasattr(self.experiment, "generate_data"):
            return None
        if not isinstance(init_key, (int, np.integer)):
            return None

        eval_key = jax.random.PRNGKey(init_key + 257)
        try:
            X_eval, y_eval = self.experiment.generate_data(eval_key)
        except TypeError:
            return None

        if X_eval.shape[0] > self.EVAL_MAX_SAMPLES:
            subset_key = jax.random.PRNGKey(init_key + 259)
            indices = jax.random.permutation(subset_key, X_eval.shape[0])[: self.EVAL_MAX_SAMPLES]
            X_eval = X_eval[indices]
            y_eval = y_eval[indices]

        return X_eval, y_eval

    def _should_save_checkpoint(self, step: int) -> bool:
        return step in self.snapshot_steps

    def _post_training_hook(self, params, results: dict) -> dict:
        results = super()._post_training_hook(params, results)
        if self.eval_ds is not None:
            X_eval, y_eval = self.eval_ds
            preds = self.model_instance(params, X_eval)
            diff = y_eval - preds
            eval_loss = 0.5 * jnp.mean(diff**2)
            results["final_eval_loss"] = float(eval_loss)
        return results

    def is_complete(self, result: dict) -> bool:
        """A run is complete if the number of loss history entries matches the expected number of steps."""
        return len(result.get("loss_history", [])) >= result.get("expected_steps", self.num_steps)


class SyntheticFixedTimeTrialRunner(SyntheticTrialRunner):
    """Trial runner for fixed-time synthetic experiments."""

    INITIAL_BATCH_KEY_SEED = 0

    # __init__ is inherited from SyntheticTrialRunner and is sufficient.

    def _init_results(self) -> dict:
        return {
            "loss_history": [],
            "batch_key_seed": self.INITIAL_BATCH_KEY_SEED,
            "expected_steps": self.num_steps,
        }

    def _create_data_iterator(self, start_step: int, results: dict) -> OnlineDataIterator:
        initial_seed = results.get("batch_key_seed", self.INITIAL_BATCH_KEY_SEED)
        return OnlineDataIterator(
            experiment=self.experiment,
            batch_size=self.run_key.batch_size,
            start_step=start_step,
            initial_batch_key_seed=initial_seed,
        )

    def _capture_iterator_state(self, data_iterator: "OnlineDataIterator", results: dict) -> dict:
        """Saves the current data generation seed to the results for checkpointing."""
        results["batch_key_seed"] = data_iterator.current_batch_key_seed
        return results


class SyntheticFixedDataTrialRunner(EpochBasedTrialRunner, SyntheticTrialRunner):
    """Trial runner for fixed-data synthetic experiments."""

    def __init__(self, context):
        super().__init__(context)
        save_epoch_snapshots = context.kwargs.get("save_epoch_snapshots", True)
        if save_epoch_snapshots:
            self._ensure_epoch_snapshot_steps(self.steps_per_epoch, self.num_epochs)

    def _init_results(self) -> dict:
        return {"loss_history": [], "epoch": 0, "expected_steps": self.num_steps}

    def _get_iterator_init_key(self) -> int:
        return self.experiment.seed

    def _on_epoch_end(self, epoch: int, params, results: dict, aux: Any) -> dict:
        results["epoch"] = epoch
        return results
