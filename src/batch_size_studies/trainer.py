import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import optax
import numpy as np

from .data_iterators import DataIterator, EpochBasedDataIterator, OnlineDataIterator
from .definitions import LossType
from .training_utils import create_base_optimizer_transform

if TYPE_CHECKING:
    from .runner import TrialContext


class DivergenceError(Exception):
    """Custom exception for when a trial's loss diverges."""


class TrialRunner(ABC):
    """Abstract base class for running a single experiment trial."""

    _JIT_CACHE = {}

    def __init__(self, context: "TrialContext"):
        self.experiment = context.experiment
        self.run_key = context.run_key
        self.params0 = context.params0
        self.model_instance = context.model_instance
        self.no_save = context.no_save

        if self.no_save:
            # If no_save is active, replace the real checkpoint manager with a no-op mock
            # that does nothing on save and returns defaults on load.
            mock_cm = Mock()
            mock_cm.load_live_checkpoint.side_effect = lambda run_key: (None, None, {}, 0)
            self.checkpoint_manager = mock_cm
        else:
            self.checkpoint_manager = context.checkpoint_manager

        self.pbar = context.pbar
        self.kwargs = context.kwargs
        self.num_steps = context.num_steps
        self.lr = self.experiment.get_adjusted_eta(self.run_key.eta)

        # --- JIT Function Caching ---
        param_shapes_pytree = jax.tree_util.tree_map(lambda x: x.shape, self.params0)
        # The pytree of shapes must be converted to a hashable type. We flatten it
        # into a tuple of leaves, treating shape tuples as leaves.
        param_shapes_tuple = tuple(
            jax.tree_util.tree_leaves(param_shapes_pytree, is_leaf=lambda x: isinstance(x, tuple))
        )

        # The cache key must uniquely identify the compiled function's structure.
        # - type(self): Differentiates between MNIST, Synthetic, etc. runners.
        # - optimizer/loss_type: These change the computation graph.
        # - id(self.model_instance): The model instance can close over `params0`
        #   (e.g., in CenteredModel). Its ID ensures we don't reuse a function
        #   with a stale `params0` from a previous sweep.
        # - param_shapes_tuple: The shapes of the parameters define the shapes
        #   for which the function is compiled.
        cache_key = (
            type(self),
            self.experiment.optimizer,
            self.experiment.loss_type,
            id(self.model_instance),  # Uniquely identify the model instance
            param_shapes_tuple,
        )

        if cache_key in self._JIT_CACHE:
            cached_funcs = self._JIT_CACHE[cache_key]
            self.loss_fn = cached_funcs["loss_fn"]
            self.jitted_update_step = cached_funcs["jitted_update_step"]
            self.base_optimizer_transform = cached_funcs["base_optimizer_transform"]
            if "eval_step" in cached_funcs:
                self.eval_step = cached_funcs["eval_step"]
        else:
            # Create functions for the first time and cache them
            self.loss_fn = self._create_loss_fn()
            self.base_optimizer_transform = create_base_optimizer_transform(self.experiment.optimizer)
            self.jitted_update_step = self._create_jitted_update_step(self.loss_fn, self.base_optimizer_transform)

            cached_funcs = {
                "loss_fn": self.loss_fn,
                "jitted_update_step": self.jitted_update_step,
                "base_optimizer_transform": self.base_optimizer_transform,
            }
            if hasattr(self, "_create_eval_step"):
                self.eval_step = self._create_eval_step()
                cached_funcs["eval_step"] = self.eval_step

            self._JIT_CACHE[cache_key] = cached_funcs

        self.opt_state_init_fn = self.base_optimizer_transform.init

    @classmethod
    def clear_cache(cls):
        """Clears the JIT function cache. Primarily for use in tests."""
        cls._JIT_CACHE.clear()

    def _check_divergence(self, loss: jnp.ndarray) -> None:
        """Checks for NaN or Inf in the loss and raises a DivergenceError if found."""
        if not jnp.isfinite(loss):
            raise DivergenceError(f"Run {self.run_key} diverged with loss: {loss}")

    def _save_checkpoint(self, step: int, params, opt_state, results: dict) -> None:
        """Saves a checkpoint if not in no_save mode."""
        self.checkpoint_manager.save_live_checkpoint(self.run_key, step, params, opt_state, results)
        # Also save a snapshot for post-hoc analysis
        self.checkpoint_manager.save_analysis_snapshot(self.run_key, step, params, self.params0)

    def run(self):
        """
        Main entry point to run the trial. This is a template method that
        orchestrates the entire lifecycle and should not be overridden.
        """
        params, loaded_opt_state, results, start_step = self.checkpoint_manager.load_live_checkpoint(self.run_key)

        # --- Compatibility fix for optimizer state ---
        # The new opt_state is a tuple `(base_state, EmptyState)`.
        # If we load an old checkpoint, we need to convert it.
        if loaded_opt_state is not None and not isinstance(loaded_opt_state, tuple):
            logging.info("Migrating old optimizer state format to new tuple format.")
            # The state for the second part of the chain (scale) is always empty.
            opt_state = (loaded_opt_state, optax.EmptyState())
        else:
            opt_state = loaded_opt_state

        # Initialize state only if no valid checkpoint was loaded
        if params is None:
            params = self.params0
            # The state for optax.chain is a tuple of states for each part of the chain.
            base_opt_state = self.opt_state_init_fn(params)
            scale_state = optax.EmptyState()
            opt_state = (base_opt_state, scale_state)
            results = self._init_results()
            start_step = 0  # Ensure start_step is 0 if no checkpoint

        # Create the data iterator, passing the correct start_step and loaded results
        data_iterator = self._create_data_iterator(start_step, results)

        try:
            final_results = self._run_training_loop(params, opt_state, results, start_step, data_iterator)
        except DivergenceError as e:
            logging.warning(e)
            return None

        return self._post_training_hook(params, final_results)

    def _run_training_loop(self, params, opt_state, results, start_step, data_iterator) -> dict:
        """Unified training loop that iterates over steps."""
        num_steps = self.num_steps

        # The enumerate start parameter correctly offsets the step counter,
        # while the iterator itself handles skipping to the right data.
        for current_step, (x_batch, y_batch) in enumerate(data_iterator, start=start_step):
            if current_step >= num_steps:
                break

            update_result = self.jitted_update_step(params, opt_state, x_batch, y_batch, self.lr)
            params, opt_state, loss, aux = update_result

            self._check_divergence(loss)
            results["loss_history"].append(loss.item())

            results = self._post_step_hook(current_step, params, results, aux)
            results = self._capture_iterator_state(data_iterator, results)

            if self._should_save_checkpoint(current_step):
                self._save_checkpoint(current_step, params, opt_state, results)

            if self.pbar:
                self.pbar.update(1)
                self.pbar.set_postfix(loss=f"{loss.item():.4f}")

        return results

    @abstractmethod
    def _init_results(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _create_loss_fn(self) -> Callable:
        raise NotImplementedError

    @abstractmethod
    def _create_jitted_update_step(
        self, loss_fn: Callable, base_optimizer_transform: optax.GradientTransformation
    ) -> Callable:
        raise NotImplementedError

    @abstractmethod
    def _create_data_iterator(self, start_step: int, results: dict) -> "DataIterator":
        """Creates the data iterator, configured to start at the correct step."""
        raise NotImplementedError

    def _post_step_hook(self, step: int, params, results: dict, aux: Any) -> dict:
        """Optional hook for actions after each training step. Returns updated results."""
        return results

    def _capture_iterator_state(self, data_iterator: "DataIterator", results: dict) -> dict:
        """Hook for subclasses to extract and save iterator state to results. Base implementation does nothing."""
        return results

    def _post_training_hook(self, params, results: dict) -> dict:
        """Optional hook for actions after the entire training loop. Returns final results."""
        return results

    def _should_save_checkpoint(self, step: int) -> bool:
        """Determines if a checkpoint should be saved at this step."""
        return False

    @abstractmethod
    def is_complete(self, result: dict) -> bool:
        """Checks if a result dictionary represents a completed run."""
        raise NotImplementedError


class MNISTTrialRunner(TrialRunner):
    """Trial runner for MNIST-based experiments."""

    EVAL_BATCH_SIZE = 512

    def __init__(self, context):
        super().__init__(context)
        self.num_epochs = context.num_epochs

        original_num_train = context.train_ds["image"].shape[0]
        self.steps_per_epoch = original_num_train // self.run_key.batch_size

        # Store context-specific data needed later
        self.test_ds = context.test_ds
        self.train_ds = context.train_ds
        self.init_key = context.init_key

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
                    loss = jnp.mean((logits - one_hot_labels) ** 2)
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
        )

    def _post_epoch_hook(self, epoch: int, params, results: dict) -> dict:
        test_accuracies = []
        num_test_samples = self.test_ds["image"].shape[0]
        for i in range((num_test_samples + self.EVAL_BATCH_SIZE - 1) // self.EVAL_BATCH_SIZE):
            start_idx = i * self.EVAL_BATCH_SIZE
            end_idx = (i + 1) * self.EVAL_BATCH_SIZE
            batch_images = self.test_ds["image"][start_idx:end_idx].reshape(-1, self.experiment.D)
            batch_labels = self.test_ds["label"][start_idx:end_idx]
            test_accuracies.append(self.eval_step(params, batch_images, batch_labels))

        epoch_accuracy = float(jnp.mean(jnp.array(test_accuracies)))
        results["epoch_test_accuracies"].append(epoch_accuracy)
        if self.pbar:
            self.pbar.set_postfix(accuracy=f"{epoch_accuracy:.4f}")

        return results

    def _step_to_completed_epoch(self, step: int) -> int:
        """Converts a training step (0-indexed) to a completed epoch number (0-indexed)."""
        return (step + 1) // self.steps_per_epoch - 1

    def _post_step_hook(self, step: int, params, results: dict, aux: Any) -> dict:
        if (step + 1) % self.steps_per_epoch == 0:
            # Calculate epoch once and use it for both description and hook
            epoch = self._step_to_completed_epoch(step)
            if self.pbar:
                self.pbar.set_description(
                    f"Sweep (B={self.run_key.batch_size}, eta={self.run_key.eta:.3g}) | Epoch {epoch + 2}/{self.num_epochs}"
                )
            results = self._post_epoch_hook(epoch, params, results)
        return results

    def _should_save_checkpoint(self, step: int) -> bool:
        # Save at the end of each epoch
        return (step + 1) % self.steps_per_epoch == 0

    def _post_training_hook(self, params, results: dict) -> dict:
        """Sets the final accuracy metric after the training loop completes."""
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
        save_dense_snapshots = context.kwargs.get("save_interstitial_snapshots", True)
        self.snapshot_steps = self._get_snapshot_steps(context.num_steps, save_dense_snapshots)
        self.eval_ds = self._create_eval_dataset(context.init_key)

    def _create_loss_fn(self) -> Callable:
        def loss_fn(params, x_batch, y_batch):
            pred = self.model_instance(params, x_batch)
            # Return a dummy aux output for a consistent interface with MNISTTrialRunner
            return jnp.mean((y_batch - pred) ** 2), None

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

        eval_key = jax.random.PRNGKey(init_key + 17)
        try:
            X_eval, y_eval = self.experiment.generate_data(eval_key)
        except TypeError:
            return None

        if X_eval.shape[0] > self.EVAL_MAX_SAMPLES:
            subset_key = jax.random.PRNGKey(init_key + 19)
            indices = jax.random.permutation(subset_key, X_eval.shape[0])[: self.EVAL_MAX_SAMPLES]
            X_eval = X_eval[indices]
            y_eval = y_eval[indices]

        return X_eval, y_eval

    def _get_snapshot_steps(self, max_steps: int, dense: bool) -> list[int]:
        """
        Generate logarithmically-spaced checkpoint steps.
        """
        if not dense:
            steps = {0}
            if max_steps > 0:
                steps.add(max_steps - 1)
            return sorted(steps)

        steps = {0}
        for magnitude in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            for base in [1, 2, 5]:
                step = base * magnitude
                if step < max_steps:
                    steps.add(step)
        if max_steps > 0:
            steps.add(max_steps - 1)
        return sorted(steps)

    def _should_save_checkpoint(self, step: int) -> bool:
        return step in self.snapshot_steps

    def _post_training_hook(self, params, results: dict) -> dict:
        results = super()._post_training_hook(params, results)
        if self.eval_ds is not None:
            X_eval, y_eval = self.eval_ds
            preds = self.model_instance(params, X_eval)
            eval_loss = jnp.mean((y_eval - preds) ** 2)
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


class SyntheticFixedDataTrialRunner(SyntheticTrialRunner):
    """Trial runner for fixed-data synthetic experiments."""

    def __init__(self, context):
        super().__init__(context)
        self.num_epochs = context.num_epochs

        original_num_train = context.train_ds[0].shape[0]
        self.steps_per_epoch = original_num_train // self.run_key.batch_size

        self.train_ds = context.train_ds

    def _init_results(self) -> dict:
        return {"loss_history": [], "epoch": 0, "expected_steps": self.num_steps}

    def _create_data_iterator(self, start_step: int, results: dict) -> EpochBasedDataIterator:
        return EpochBasedDataIterator(
            train_ds=self.train_ds,
            batch_size=self.run_key.batch_size,
            num_epochs=self.num_epochs,
            init_key=self.experiment.seed,
            start_step=start_step,
        )

    def _step_to_completed_epoch(self, step: int) -> int:
        """Converts a training step (0-indexed) to a completed epoch number (0-indexed)."""
        return (step + 1) // self.steps_per_epoch - 1

    def _post_step_hook(self, step: int, params, results: dict, aux: Any) -> dict:
        """Saves the current epoch number to the results for checkpointing."""
        if (step + 1) % self.steps_per_epoch == 0:
            # Called at the end of an epoch, so step+1 has completed an epoch.
            epoch = self._step_to_completed_epoch(step)
            results["epoch"] = epoch
        return results
