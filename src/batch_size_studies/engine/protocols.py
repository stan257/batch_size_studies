"""
This module defines the core abstract base classes (protocols) that decouple
different parts of the framework, such as the experiment definitions from the
concrete trial runner implementations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generator, List, Protocol
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .training_utils import create_base_optimizer_transform

if TYPE_CHECKING:
    from .runner import TrialContext


class DataIterator(ABC):
    """Abstract base class for data iterators."""

    @abstractmethod
    def __iter__(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yields batches of (inputs, targets)."""
        raise NotImplementedError


@dataclass(frozen=True)
class TrainingOptions:
    max_eval_samples: int | None = None
    save_interstitial_snapshots: bool = False
    save_epoch_snapshots: bool = True
    disable_eval_dataset: bool = False


class ModelProtocol(Protocol):
    def init_params(self, init_key: int, widths: List[int]) -> Any: ...

    def __call__(self, params: Any, inputs: jnp.ndarray) -> jnp.ndarray: ...


class DivergenceError(Exception):
    """Custom exception for when a trial's loss diverges."""


class TrialRunner(ABC):
    """
    Template that drives a single (B, η) training run end-to-end: it builds the loss,
    runs the step loop, records metrics, and decides when to checkpoint/snapshot.
    Subclasses only supply data iterators and per-experiment hooks while reusing this
    orchestration logic. Jitted update/eval functions are cached at class scope; call
    TrialRunner.clear_cache() in interactive sessions if you redefine models or losses.
    """

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
        provided_options = context.options if hasattr(context, "options") else None
        if isinstance(provided_options, TrainingOptions):
            self.options = provided_options
        else:
            self.options = TrainingOptions(
                max_eval_samples=self.kwargs.get("max_eval_samples"),
                save_interstitial_snapshots=bool(self.kwargs.get("save_interstitial_snapshots", False)),
                save_epoch_snapshots=bool(self.kwargs.get("save_epoch_snapshots", True)),
                disable_eval_dataset=bool(self.kwargs.get("disable_eval_dataset", False)),
            )
        self.num_steps = context.num_steps
        self.lr = self.experiment.get_adjusted_eta(self.run_key.eta)

        # JIT function caching
        param_shapes_pytree = jax.tree_util.tree_map(lambda x: x.shape, self.params0)
        param_shapes_tuple = tuple(
            jax.tree_util.tree_leaves(param_shapes_pytree, is_leaf=lambda x: isinstance(x, tuple))
        )
        cache_key = (
            type(self),
            self.experiment.optimizer,
            self.experiment.loss_type,
            id(self.model_instance),
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
        self.checkpoint_manager.save_analysis_snapshot(self.run_key, step, params, self.params0)

    def run(self):
        """
        Main entry point to run the trial. This is a template method that
        orchestrates the entire lifecycle and should not be overridden.
        """
        params, loaded_opt_state, results, start_step = self.checkpoint_manager.load_live_checkpoint(self.run_key)

        if loaded_opt_state is not None and not isinstance(loaded_opt_state, tuple):
            logging.info("Migrating old optimizer state format to new tuple format.")
            opt_state = (loaded_opt_state, optax.EmptyState())
        else:
            opt_state = loaded_opt_state

        if params is None:
            params = self.params0
            base_opt_state = self.opt_state_init_fn(params)
            scale_state = optax.EmptyState()
            opt_state = (base_opt_state, scale_state)
            results = self._init_results()
            start_step = 0

        start_step = self._adjust_start_step(start_step, results)
        data_iterator = self._create_data_iterator(start_step, results)

        try:
            params, opt_state, final_results = self._run_training_loop(
                params, opt_state, results, start_step, data_iterator
            )
        except DivergenceError as e:
            logging.warning(e)
            return None

        return self._post_training_hook(params, final_results)

    def _run_training_loop(self, params, opt_state, results, start_step, data_iterator):
        """Unified training loop that iterates over steps."""
        for current_step, (x_batch, y_batch) in enumerate(data_iterator, start=start_step):
            if current_step >= self.num_steps:
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

        return params, opt_state, results

    @abstractmethod
    def _init_results(self) -> dict: ...

    @abstractmethod
    def _create_loss_fn(self) -> Callable: ...

    @abstractmethod
    def _create_jitted_update_step(
        self, loss_fn: Callable, base_optimizer_transform: optax.GradientTransformation
    ) -> Callable: ...

    @abstractmethod
    def _create_data_iterator(self, start_step: int, results: dict) -> "DataIterator": ...

    def _adjust_start_step(self, start_step: int, results: dict):
        """Allows subclasses to override how resume steps are resolved."""
        return start_step

    def _post_step_hook(self, step: int, params, results: dict, aux: Any) -> dict:
        """Optional hook for actions after each training step. Returns updated results."""
        return results

    def _capture_iterator_state(self, data_iterator: "DataIterator", results: dict) -> dict:
        """Hook for subclasses to extract and save iterator state. Base implementation does nothing."""
        return results

    def _post_training_hook(self, params, results: dict) -> dict:
        """Optional hook for actions after the entire training loop. Returns final results."""
        return results

    def _should_save_checkpoint(self, step: int) -> bool:
        """Determines if a checkpoint should be saved at this step."""
        return False

    @staticmethod
    def _log_spaced_steps(max_steps: int) -> set[int]:
        steps: set[int] = set()
        magnitude = 1
        while magnitude < max_steps:
            for base in (1, 2, 5):
                step = base * magnitude
                if 0 < step < max_steps:
                    steps.add(step)
            magnitude *= 10
        return steps

    def _compute_snapshot_steps(self, max_steps: int, dense: bool) -> list[int]:
        steps: set[int] = {0}
        if max_steps > 0:
            steps.add(max_steps - 1)
        if dense:
            steps |= self._log_spaced_steps(max_steps)
        return sorted(steps)

    def _ensure_epoch_snapshot_steps(self, steps_per_epoch: int | None, num_epochs: int | None):
        if steps_per_epoch is None or steps_per_epoch <= 0:
            return
        if num_epochs is None or num_epochs <= 0:
            num_epochs = max(1, (self.num_steps + steps_per_epoch - 1) // steps_per_epoch)

        epoch_steps = set()
        for epoch_idx in range(num_epochs):
            step = (epoch_idx + 1) * steps_per_epoch - 1
            if 0 <= step < self.num_steps:
                epoch_steps.add(step)
        if not epoch_steps:
            return

        combined = set(getattr(self, "snapshot_steps", [])) | epoch_steps
        self.snapshot_steps = sorted(combined)

    @abstractmethod
    def is_complete(self, result: dict) -> bool: ...
