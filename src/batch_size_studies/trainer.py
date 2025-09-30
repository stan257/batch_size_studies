import logging
from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .checkpoint_utils import CheckpointManager
from .definitions import LossType, RunKey
from .models import MLP
from .training_utils import create_optimizer


class TrialRunner(ABC):
    """Abstract base class for running a single experiment trial."""

    def __init__(
        self,
        experiment,
        run_key: RunKey,
        params0,
        mlp_instance: MLP,
        checkpoint_manager: CheckpointManager,
        pbar,
        no_save: bool,
        **kwargs,
    ):
        self.experiment = experiment
        self.run_key = run_key
        self.params0 = params0
        self.mlp_instance = mlp_instance
        self.checkpoint_manager = checkpoint_manager
        self.pbar = pbar
        self.no_save = no_save
        self.kwargs = kwargs

        self.optimizer = create_optimizer(self.experiment, self.run_key.eta)
        self.loss_fn = self._create_loss_fn()
        self.update_step = self._create_update_step()

    def _check_divergence(self, loss: jnp.ndarray) -> bool:
        """Checks for NaN or Inf in the loss and logs a warning."""
        if not jnp.isfinite(loss):
            logging.warning(f"Run {self.run_key} diverged. Stopping trial.")
            return True
        return False

    def _save_checkpoint(self, step: int, params, opt_state, results: dict):
        """Saves a checkpoint if not in no_save mode."""
        if not self.no_save:
            self.checkpoint_manager.save_live_checkpoint(self.run_key, step, params, opt_state, results)
            # Also save a snapshot for post-hoc analysis
            self.checkpoint_manager.save_analysis_snapshot(self.run_key, step, params, self.params0)

    def run(self):
        """Main entry point to run the trial."""
        if self.no_save:
            params, opt_state, results, start_marker = None, None, self._init_results(), 0
        else:
            params, opt_state, results, start_marker = self.checkpoint_manager.load_live_checkpoint(self.run_key)

        if params is None:
            params = self.params0
            opt_state = self.optimizer.init(params)
            results = self._init_results()

        return self._run_training_loop(params, opt_state, results, start_marker)

    @abstractmethod
    def _init_results(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _create_loss_fn(self):
        raise NotImplementedError

    @abstractmethod
    def _create_update_step(self):
        raise NotImplementedError

    @abstractmethod
    def _run_training_loop(self, params, opt_state, results, start_marker) -> dict | None:
        raise NotImplementedError


class EpochBasedTrialRunner(TrialRunner):
    """Abstract base class for epoch-based training loops."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_epochs = self.kwargs.get("num_epochs", getattr(self.experiment, "num_epochs", 1))
        self.num_train = 0  # Subclasses must set this

    @abstractmethod
    def _get_batch_generator(self, epoch: int) -> partial:
        """Yields batches of (inputs, targets) for a given epoch."""
        raise NotImplementedError

    def _post_epoch_hook(self, epoch: int, params, results: dict) -> dict:
        """Optional hook for end-of-epoch actions like evaluation. Returns updated results."""
        return results

    def _should_save_checkpoint(self, step_type: str, step_value: int) -> bool:
        """Determines if a checkpoint should be saved at this step/epoch."""
        # Default behavior: save at the end of each epoch.
        return step_type == "epoch"

    def _run_training_loop(self, params, opt_state, results, start_marker) -> dict | None:
        start_epoch = start_marker
        if self.num_train == 0:
            raise NotImplementedError(f"{self.__class__.__name__} must set the 'num_train' attribute.")

        steps_per_epoch = self.num_train // self.run_key.batch_size
        current_step = start_epoch * steps_per_epoch

        for epoch in range(start_epoch, self.num_epochs):
            self.pbar.set_description(
                f"Sweep (B={self.run_key.batch_size}, eta={self.run_key.eta:.3g}) | Epoch {epoch + 1}/{self.num_epochs}"
            )
            batch_generator = self._get_batch_generator(epoch)

            for x_batch, y_batch in batch_generator:
                update_result = self.update_step(params, opt_state, x_batch, y_batch)
                params, opt_state, loss = update_result[0], update_result[1], update_result[2]

                if self._check_divergence(loss):
                    return None
                results["loss_history"].append(loss.item())

                if self._should_save_checkpoint("step", current_step):
                    self._save_checkpoint(current_step, params, opt_state, results)
                current_step += 1

            results = self._post_epoch_hook(epoch, params, results)

            if self._should_save_checkpoint("epoch", epoch):
                self._save_checkpoint(epoch, params, opt_state, results)

        return results


class MNISTTrialRunner(EpochBasedTrialRunner):
    """Trial runner for MNIST-based experiments."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_ds = self.kwargs["train_ds"]
        self.test_ds = self.kwargs["test_ds"]
        self.init_key = self.kwargs["init_key"]
        self.eval_step = self._create_eval_step()
        self.num_train = self.train_ds["image"].shape[0]

    def _init_results(self) -> dict:
        return {"epoch_test_accuracies": [], "loss_history": []}

    def _create_loss_fn(self):
        apply_fn = jax.jit(self.mlp_instance)
        match self.experiment.loss_type:
            case LossType.XENT:

                def loss_fn(params, x_batch, y_batch_labels):
                    logits = apply_fn(params, x_batch) - apply_fn(self.params0, x_batch)
                    one_hot_labels = jax.nn.one_hot(y_batch_labels, num_classes=self.experiment.num_outputs)
                    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels))
                    return loss, logits

                return loss_fn
            case LossType.MSE:

                def loss_fn(params, x_batch, y_batch_labels):
                    logits = apply_fn(params, x_batch) - apply_fn(self.params0, x_batch)
                    one_hot_labels = jax.nn.one_hot(y_batch_labels, num_classes=self.experiment.num_outputs)
                    loss = jnp.mean((logits - one_hot_labels) ** 2)
                    return loss, logits

                return loss_fn
            case _:
                raise NotImplementedError(f"Loss type {self.experiment.loss_type} not implemented.")

    def _create_update_step(self):
        @jax.jit
        def update_step(params, opt_state, x_batch, y_batch):
            (loss, logits), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(params, x_batch, y_batch)
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            accuracy = jnp.mean(jnp.argmax(logits, -1) == y_batch)
            return new_params, new_opt_state, loss, accuracy

        return update_step

    def _create_eval_step(self):
        apply_fn = jax.jit(self.mlp_instance)

        @jax.jit
        def eval_step(params, x_batch, y_batch):
            logits = apply_fn(params, x_batch) - apply_fn(self.params0, x_batch)
            accuracy = jnp.mean(jnp.argmax(logits, -1) == y_batch)
            return accuracy

        return eval_step

    def _get_batch_generator(self, epoch: int) -> partial:
        num_steps_per_epoch = self.num_train // self.run_key.batch_size
        rng = jr.PRNGKey(self.init_key + epoch + 1)
        perms = jr.permutation(rng, self.num_train)[: num_steps_per_epoch * self.run_key.batch_size]
        perms = perms.reshape((num_steps_per_epoch, self.run_key.batch_size))
        np_perms = np.array(perms)

        for perm in np_perms:
            batch_images = self.train_ds["image"][perm, ...].reshape(self.run_key.batch_size, -1)
            batch_labels = self.train_ds["label"][perm, ...]
            yield batch_images, batch_labels

    def _post_epoch_hook(self, epoch: int, params, results: dict) -> dict:
        test_accuracies = []
        num_test, eval_batch_size = self.test_ds["image"].shape[0], 512
        for i in range((num_test + eval_batch_size - 1) // eval_batch_size):
            start_idx, end_idx = i * eval_batch_size, (i + 1) * eval_batch_size
            batch_images = self.test_ds["image"][start_idx:end_idx].reshape(-1, self.experiment.D)
            batch_labels = self.test_ds["label"][start_idx:end_idx]
            if batch_images.shape[0] > 0:
                test_accuracies.append(self.eval_step(params, batch_images, batch_labels))

        epoch_accuracy = float(jnp.mean(jnp.array(test_accuracies)))
        results["epoch_test_accuracies"].append(epoch_accuracy)
        self.pbar.set_postfix(accuracy=f"{epoch_accuracy:.4f}")

        return results

    def _run_training_loop(self, params, opt_state, results, start_epoch) -> dict | None:
        # This is now a thin wrapper to call the base class implementation
        # and then set the final accuracy metric.
        results = super()._run_training_loop(params, opt_state, results, start_epoch)

        if results is None:
            return None

        if results.get("epoch_test_accuracies"):
            results["final_test_accuracy"] = results["epoch_test_accuracies"][-1]
        return results


class SyntheticTrialRunner(TrialRunner):
    """Base trial runner for synthetic data experiments."""

    def _create_loss_fn(self):
        def loss_fn(params, x_batch, y_batch):
            pred = self.mlp_instance(params, x_batch) - self.mlp_instance(self.params0, x_batch)
            return jnp.mean((y_batch - pred) ** 2)

        return partial(loss_fn)

    def _create_update_step(self):
        @jax.jit
        def update_step(params, opt_state, x_batch, y_batch):
            loss, grad = jax.value_and_grad(self.loss_fn)(params, x_batch, y_batch)
            updates, new_opt_state = self.optimizer.update(grad, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        return update_step

    def _get_snapshot_steps(self, max_steps: int) -> list[int]:
        steps = {0}
        for magnitude in [1, 10, 100, 1000, 10000, 100000, 1000000]:
            for base in [1, 2, 5]:
                step = base * magnitude
                if step < max_steps:
                    steps.add(step)
        if max_steps > 0:
            steps.add(max_steps - 1)
        return sorted(list(steps))


class SyntheticFixedTimeTrialRunner(SyntheticTrialRunner):  # Does not use EpochBasedTrialRunner
    """Trial runner for fixed-time synthetic experiments."""

    def _init_results(self) -> dict:
        return {"loss_history": [], "batch_key_seed": 0}

    def _run_training_loop(self, params, opt_state, results, start_step) -> dict | None:
        num_steps = self.kwargs["num_steps"]
        batch_key_seed = results.get("batch_key_seed", 0)
        step_for_curr_data = 0
        X_data, y_data = self.experiment.generate_data(jr.key(batch_key_seed))
        snapshot_steps = self._get_snapshot_steps(num_steps)

        for current_step in range(start_step, num_steps):
            if (step_for_curr_data + 1) * self.run_key.batch_size > self.experiment.P:
                batch_key_seed += 1
                X_data, y_data = self.experiment.generate_data(jr.key(batch_key_seed))
                step_for_curr_data = 0

            start = step_for_curr_data * self.run_key.batch_size
            X_batch, y_batch = (
                X_data[start : start + self.run_key.batch_size],
                y_data[start : start + self.run_key.batch_size],
            )

            params, opt_state, loss, *_ = self.update_step(params, opt_state, X_batch, y_batch)
            step_for_curr_data += 1

            if self._check_divergence(loss):
                return None
            results["loss_history"].append(loss.item())
            results["batch_key_seed"] = batch_key_seed

            if current_step in snapshot_steps:
                self._save_checkpoint(current_step, params, opt_state, results)

            if self.pbar:
                self.pbar.set_postfix(loss=f"{loss.item():.4f}", step=f"{current_step + 1}/{num_steps}")
        return results


class SyntheticFixedDataTrialRunner(EpochBasedTrialRunner, SyntheticTrialRunner):
    """Trial runner for fixed-data synthetic experiments."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X_data = self.kwargs["X_data"]
        self.y_data = self.kwargs["y_data"]
        self.num_train = self.X_data.shape[0]
        self.snapshot_steps = self._get_snapshot_steps(self.num_epochs * (self.num_train // self.run_key.batch_size))

    def _init_results(self) -> dict:
        return {"loss_history": [], "epoch": 0}

    def _get_batch_generator(self, epoch: int) -> partial:
        steps_per_epoch = self.num_train // self.run_key.batch_size
        rng = jr.PRNGKey(getattr(self.experiment, "seed", 0) + epoch)
        perms = jr.permutation(rng, self.num_train)[: steps_per_epoch * self.run_key.batch_size]
        epoch_perms = perms.reshape((steps_per_epoch, self.run_key.batch_size))

        for perm in epoch_perms:
            yield self.X_data[perm, ...], self.y_data[perm, ...]

    def _post_epoch_hook(self, epoch: int, params, results: dict) -> dict:
        results["epoch"] = epoch
        return results

    def _should_save_checkpoint(self, step_type: str, step_value: int) -> bool:
        return step_type == "step" and step_value in self.snapshot_steps
