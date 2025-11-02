import logging
from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import optax

from .data_iterators import EpochBasedDataIterator, OnlineDataIterator
from .definitions import LossType
from .training_utils import create_optimizer


class TrialRunner(ABC):
    """Abstract base class for running a single experiment trial."""

    def __init__(self, context):
        self.experiment = context.experiment
        self.run_key = context.run_key
        self.params0 = context.params0
        self.model_instance = context.model_instance
        self.checkpoint_manager = context.checkpoint_manager
        self.pbar = context.pbar
        self.no_save = context.no_save
        self.kwargs = context.kwargs
        self.num_steps = context.num_steps
        self.lr = self.experiment.get_adjusted_eta(self.run_key.eta)
        # Subclasses are now responsible for creating these
        self.optimizer, self.loss_fn, self.update_step, self.data_iterator = None, None, None, None

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
            params, opt_state, results, start_step = None, None, self._init_results(), 0
        else:
            # load_live_checkpoint now consistently returns a start_step
            params, opt_state, results, start_step = self.checkpoint_manager.load_live_checkpoint(self.run_key)

        if params is None:
            params = self.params0
            opt_state = self.optimizer.init(params)
            results = self._init_results()
            start_step = 0  # Ensure start_step is 0 if no checkpoint

        return self._run_training_loop(params, opt_state, results, start_step)

    def _run_training_loop(self, params, opt_state, results, start_step) -> dict | None:
        """Unified training loop that iterates over steps."""
        num_steps = self.num_steps

        for current_step, (x_batch, y_batch) in enumerate(self.data_iterator, start=start_step):
            if current_step >= num_steps:
                break

            update_result = self.update_step(params, opt_state, x_batch, y_batch)
            params, opt_state, loss = update_result[0], update_result[1], update_result[2]

            if self._check_divergence(loss):
                return None
            results["loss_history"].append(loss.item())

            results = self._post_step_hook(current_step, params, results)

            if self._should_save_checkpoint(current_step):
                self._save_checkpoint(current_step, params, opt_state, results)

            if self.pbar:
                self.pbar.set_postfix(loss=f"{loss.item():.4f}", step=f"{current_step + 1}/{num_steps}")

        return self._post_training_hook(params, results)

    @abstractmethod
    def _init_results(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _create_loss_fn(self):
        raise NotImplementedError

    @abstractmethod
    def _create_update_step(self):
        raise NotImplementedError

    def _post_step_hook(self, step: int, params, results: dict) -> dict:
        """Optional hook for actions after each training step. Returns updated results."""
        return results

    def _post_training_hook(self, params, results: dict) -> dict | None:
        """Optional hook for actions after the entire training loop. Returns final results."""
        return results

    def _should_save_checkpoint(self, step: int) -> bool:
        """Determines if a checkpoint should be saved at this step."""
        return False


class MNISTTrialRunner(TrialRunner):
    """Trial runner for MNIST-based experiments."""

    def __init__(self, context):
        super().__init__(context)
        self.num_epochs = context.kwargs.get("num_epochs", getattr(self.experiment, "num_epochs", 1))

        self.optimizer = create_optimizer(self.experiment, self.lr)
        self.loss_fn = self._create_loss_fn()
        self.update_step = self._create_update_step()
        self.eval_step = self._create_eval_step()

        original_num_train = context.train_ds["image"].shape[0]
        self.steps_per_epoch = original_num_train // self.run_key.batch_size

        self.data_iterator = EpochBasedDataIterator(
            train_ds=context.train_ds,
            batch_size=self.run_key.batch_size,
            num_epochs=self.num_epochs,
            init_key=context.init_key,
        )
        self.test_ds = context.test_ds

    def _init_results(self) -> dict:
        return {"epoch_test_accuracies": [], "loss_history": []}

    def _create_loss_fn(self):
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
        apply_fn = self.model_instance

        @jax.jit
        def eval_step(params, x_batch, y_batch):
            logits = apply_fn(params, x_batch)
            accuracy = jnp.mean(jnp.argmax(logits, -1) == y_batch)
            return accuracy

        return eval_step

    def run(self):
        """Overrides run to set the correct start_step on the data iterator."""
        if self.no_save:
            _, _, _, start_step = None, None, self._init_results(), 0
        else:
            _, _, _, start_step = self.checkpoint_manager.load_live_checkpoint(self.run_key)

        self.data_iterator.start_step = start_step
        return super().run()

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

    def _post_step_hook(self, step: int, params, results: dict) -> dict:
        if (step + 1) % self.steps_per_epoch == 0:
            epoch = (step + 1) // self.steps_per_epoch - 1
            if self.pbar:
                self.pbar.set_description(
                    f"Sweep (B={self.run_key.batch_size}, eta={self.run_key.eta:.3g}) | Epoch {epoch + 2}/{self.num_epochs}"
                )
            epoch = (step + 1) // self.steps_per_epoch - 1
            results = self._post_epoch_hook(epoch, params, results)
        return results

    def _should_save_checkpoint(self, step: int) -> bool:
        # Save at the end of each epoch
        return (step + 1) % self.steps_per_epoch == 0

    def _post_training_hook(self, params, results: dict) -> dict | None:
        """Sets the final accuracy metric after the training loop completes."""
        if results is None:  # Handle divergence case
            return None

        if results.get("epoch_test_accuracies"):
            results["final_test_accuracy"] = results["epoch_test_accuracies"][-1]
        return results


class SyntheticTrialRunner(TrialRunner):
    """Base trial runner for synthetic data experiments."""

    def __init__(self, context):
        super().__init__(context)
        # These are defined here because they are common to all synthetic runners
        self.optimizer = create_optimizer(self.experiment, self.lr)
        self.loss_fn = self._create_loss_fn()
        self.update_step = self._create_update_step()

    def _create_loss_fn(self):
        def loss_fn(params, x_batch, y_batch):
            pred = self.model_instance(params, x_batch)
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


class SyntheticFixedTimeTrialRunner(SyntheticTrialRunner):
    """Trial runner for fixed-time synthetic experiments."""

    def __init__(self, context):
        super().__init__(context)
        self.snapshot_steps = self._get_snapshot_steps(context.num_steps)
        # The data_iterator is created inside the `run` method,
        # once the checkpoint state is loaded.
        self.data_iterator = None

    def _init_results(self) -> dict:
        return {"loss_history": [], "batch_key_seed": 0}

    def run(self):
        """Overrides run to create the data iterator with the correct start_step and results."""
        if self.no_save:
            params, opt_state, results, start_step = None, None, self._init_results(), 0
        else:
            params, opt_state, results, start_step = self.checkpoint_manager.load_live_checkpoint(self.run_key)

        # Create the data iterator here, now that we have the loaded state.
        initial_seed = results.get("batch_key_seed", 0)
        self.data_iterator = OnlineDataIterator(
            experiment=self.experiment,
            batch_size=self.run_key.batch_size,
            start_step=start_step,
            initial_batch_key_seed=initial_seed,
        )
        return super().run()

    def _post_step_hook(self, step: int, params, results: dict) -> dict:
        """Saves the current data generation seed to the results for checkpointing."""
        results["batch_key_seed"] = self.data_iterator.current_batch_key_seed
        return results

    def _should_save_checkpoint(self, step: int) -> bool:
        return step in self.snapshot_steps


class SyntheticFixedDataTrialRunner(SyntheticTrialRunner):
    """Trial runner for fixed-data synthetic experiments."""

    def __init__(self, context):
        super().__init__(context)
        self.num_epochs = context.kwargs.get("num_epochs", getattr(self.experiment, "num_epochs", 1))

        original_num_train = context.train_ds[0].shape[0]
        self.steps_per_epoch = original_num_train // self.run_key.batch_size

        self.data_iterator = EpochBasedDataIterator(
            train_ds=context.train_ds,
            batch_size=self.run_key.batch_size,
            num_epochs=self.num_epochs,
            init_key=getattr(self.experiment, "seed", 0),
        )

        self.snapshot_steps = self._get_snapshot_steps(self.num_epochs * self.steps_per_epoch)

    def _init_results(self) -> dict:
        return {"loss_history": [], "epoch": 0}

    def run(self):
        """Overrides run to set the correct start_step on the data iterator."""
        _, _, _, start_step = self.checkpoint_manager.load_live_checkpoint(self.run_key)
        self.data_iterator.start_step = start_step
        return super().run()

    def _post_step_hook(self, step: int, params, results: dict) -> dict:
        """Saves the current epoch number to the results for checkpointing."""
        if (step + 1) % self.steps_per_epoch == 0:
            epoch = (step + 1) // self.steps_per_epoch - 1
            results["epoch"] = epoch
        return results

    def _should_save_checkpoint(self, step: int) -> bool:
        return step in self.snapshot_steps
