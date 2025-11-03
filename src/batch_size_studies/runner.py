"""
Unified Training Runner

This module provides a single, unified entry point for running all types of
hyperparameter sweeps (synthetic, MNIST, etc.). It centralizes the logic for
looping over hyperparameters, managing checkpoints, and saving results, while
dispatching to type-specific trial runners.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from tqdm.auto import tqdm

from .checkpoint_utils import CheckpointManager
from .definitions import ModelProtocol, RunKey
from .paths import EXPERIMENTS_DIR


@dataclass
class TrialContext:
    """Encapsulates all configuration and data for a single trial."""

    experiment: Any
    run_key: RunKey
    params0: Any
    model_instance: Any
    checkpoint_manager: CheckpointManager
    pbar: tqdm
    no_save: bool
    init_key: int
    num_steps: int
    num_epochs: int
    kwargs: dict = field(default_factory=dict)
    # Data fields, can be None
    train_ds: Any | None = None
    test_ds: Any | None = None


def _subsample_dataset(dataset: dict, max_samples: int, key: jax.random.PRNGKey) -> dict:
    """Helper to subsample a dictionary-based dataset."""
    num_original_samples = len(dataset["image"])
    if num_original_samples > max_samples:
        indices_to_use = jax.random.permutation(key, num_original_samples)[:max_samples]
        subsampled_dataset = {
            "image": dataset["image"][np.array(indices_to_use)],
            "label": dataset["label"][np.array(indices_to_use)],
        }
        logging.info(f"Evaluating on a fixed random subset of {len(subsampled_dataset['image'])} test samples.")
        return subsampled_dataset
    return dataset


@dataclass
class RunStatus:
    """Determines if a given trial should be run or skipped."""

    run_key: RunKey
    results_dict: dict
    failed_runs: set
    num_steps: int
    no_save: bool

    @property
    def is_successful(self) -> bool:
        """Checks if the run has already been completed successfully."""
        if self.no_save or self.run_key not in self.results_dict:
            return False
        result = self.results_dict.get(self.run_key, {})
        return self.run_key in self.results_dict and "loss_history" in result

    @property
    def should_run(self) -> bool:
        """Determines if the trial should be executed."""
        if self.no_save:
            return True
        if self.run_key in self.failed_runs:
            logging.info(f"Skipping previously failed run {self.run_key}")
            return False

        # A run is only skipped if it's fully complete with respect to the *current* num_steps
        result = self.results_dict.get(self.run_key, {})
        loss_history = result.get("loss_history", [])
        is_fully_complete = len(loss_history) >= self.num_steps
        if is_fully_complete:
            logging.info(f"Skipping completed run {self.run_key}")
            return False
        return True


class EtaStabilityTracker:
    """Tracks consecutive successful runs to enable early stopping of an eta sweep."""

    def __init__(self, depth: int | None):
        self.depth = depth
        self.count = 0

    def update(self, is_successful: bool) -> bool:
        """Updates the tracker and returns True if the stopping condition is met."""
        if self.depth is None or self.depth <= 0:
            return False

        if is_successful:
            self.count += 1
        else:
            self.count = 0

        if self.count >= self.depth:
            logging.info(f"Found {self.depth} consecutive stable etas. Skipping remaining etas for this batch size.")
            return True
        return False

    def reset(self):
        """Resets the counter."""
        self.count = 0


# ============================================================================
# INITIALIZATION HELPERS
# ============================================================================


def initialize_results_and_checkpoints(experiment, directory: str, no_save: bool):
    """Initializes results, failed runs, and the checkpoint manager."""
    if no_save:
        results_dict, failed_runs = {}, set()
    else:
        results_dict, failed_runs = experiment.load_results(directory=directory, silent=True)

    checkpoint_manager = CheckpointManager(experiment, directory=directory)
    return results_dict, failed_runs, checkpoint_manager


def initialize_model_params(
    model_instance: ModelProtocol,
    checkpoint_manager: CheckpointManager,
    init_key: int,
    widths: list[int],
    no_save: bool,
):
    """Initializes or loads the initial model parameters (params0)."""
    if no_save:
        return model_instance.init_params(init_key, widths)

    # This method handles both loading and safe, locked initialization.
    return checkpoint_manager.initialize_and_save_initial_params(init_key, model_instance, widths)


# ============================================================================
# RESULT VALIDATION AND STORAGE
# ============================================================================


def _validate_and_store_partial_result(
    result: dict | None,
    run_key: RunKey,
    results_dict: dict,
    failed_runs: set,
    experiment,
    no_save: bool,
    directory: str,
) -> bool:
    """
    Validates a trial's result and updates tracking dictionaries.

    A result is "valid" if it's not None and contains finite metrics.
    """
    is_valid = result is not None
    if "final_test_accuracy" in (result or {}) and not np.isfinite(result["final_test_accuracy"]):
        is_valid = False

    if is_valid:
        # A valid result is always stored, even if incomplete.
        # This allows for resumption.
        results_dict[run_key] = result
        failed_runs.discard(run_key)  # Remove from failed set if it previously failed
    else:
        # An invalid result (e.g., from divergence) means the run has failed.
        # The previous partial result (if any) is removed.
        failed_runs.add(run_key)
        results_dict.pop(run_key, None)

    if not no_save:
        # Save results after every trial
        experiment.save_results(results_dict, failed_runs, directory)

    return is_valid


# ============================================================================
# TRIAL EXECUTION HELPERS
# ============================================================================


def run_single_trial(
    context: TrialContext,
    results_dict: dict,
    failed_runs: set,
) -> bool:
    """
    Checks the status of, runs, and validates a single trial configuration.
    Returns True if the run was successful (or already was), False otherwise.
    """
    status = RunStatus(context.run_key, results_dict, failed_runs, context.num_steps, context.no_save)

    if not status.should_run:
        # If we skip, it's because it's already complete for this context.
        return True

    trial_runner = get_trial_runner(context)

    if trial_runner:
        result = trial_runner.run()
        is_valid = _validate_and_store_partial_result(
            result,
            context.run_key,
            results_dict,
            failed_runs,
            context.experiment,
            context.no_save,
            context.checkpoint_manager.directory,
        )
        is_complete = is_valid and trial_runner.is_complete(result)
        if is_complete and not context.no_save:
            context.checkpoint_manager.cleanup_live_checkpoint(context.run_key)
    else:
        logging.error(f"Could not create a trial runner for {context.run_key}. Marking as failed.")
        failed_runs.add(context.run_key)
        is_complete = False

    return is_complete


# ============================================================================
# SWEEP ORCHESTRATION HELPERS
# ============================================================================


def _setup_sweep_state(experiment, directory, no_save, init_key):
    """Handles all initial setup for a sweep."""
    results_dict, failed_runs, checkpoint_manager = initialize_results_and_checkpoints(experiment, directory, no_save)

    model_instance = experiment.create_model_instance()
    widths = experiment.get_model_widths()
    params0 = initialize_model_params(model_instance, checkpoint_manager, init_key, widths, no_save)
    model_for_runner = experiment.get_model_wrapper(model_instance, params0)

    return results_dict, failed_runs, checkpoint_manager, params0, model_for_runner


def _execute_sweep_loops(
    experiment,
    batch_sizes,
    etas,
    init_key,
    no_save,
    eta_stability_search_depth,
    results_dict,
    failed_runs,
    checkpoint_manager,
    params0,
    model_for_runner,
    train_ds,
    test_ds,
    **kwargs,
):
    """Contains the main nested loops for the hyperparameter sweep."""
    for batch_size in tqdm(batch_sizes, desc="Batch Size Sweep"):
        if experiment.should_skip_batch_size(batch_size, train_ds):
            continue

        eta_tracker = EtaStabilityTracker(eta_stability_search_depth)
        sorted_etas = sorted(etas, reverse=True)

        for eta in tqdm(sorted_etas, desc=f"Eta Sweep (B={batch_size})", leave=False):
            run_key = RunKey(batch_size=batch_size, eta=eta)
            num_epochs_override = kwargs.get("num_epochs")
            num_steps, num_epochs_for_run = experiment.compute_num_steps(
                batch_size, train_ds, num_epochs=num_epochs_override
            )

            status = RunStatus(run_key, results_dict, failed_runs, num_steps, no_save)
            if not status.should_run:
                is_successful = status.is_successful
                if eta_tracker.update(is_successful):
                    break
                continue

            # Create and manage the lifecycle of the progress bar for active trials
            with tqdm(total=num_steps, desc=f"η={eta:.3g}", leave=False) as pbar:
                context = TrialContext(
                    experiment=experiment,
                    run_key=run_key,
                    params0=params0,
                    model_instance=model_for_runner,
                    checkpoint_manager=checkpoint_manager,
                    train_ds=train_ds,
                    test_ds=test_ds,
                    pbar=pbar,
                    no_save=no_save,
                    init_key=init_key,
                    num_steps=num_steps,
                    num_epochs=num_epochs_for_run,
                    kwargs=kwargs,
                )
                is_successful = run_single_trial(context=context, results_dict=results_dict, failed_runs=failed_runs)

            if eta_tracker.update(is_successful):
                break


def run_experiment_sweep(
    experiment,
    batch_sizes: list[int],
    etas: list[float],
    init_key: int = 0,
    directory=EXPERIMENTS_DIR,
    no_save: bool = False,
    eta_stability_search_depth: int | None = None,
    max_eval_samples: int | None = None,
    **kwargs,
):
    """
    Orchestrates a full hyperparameter sweep, dispatching to the correct
    training logic based on the type of the experiment object.
    """
    # 1. Setup
    results_dict, failed_runs, checkpoint_manager, params0, model_for_runner = _setup_sweep_state(
        experiment, directory, no_save, init_key
    )

    # 2. Load Data
    train_ds, test_ds = experiment.prepare_datasets(init_key, **kwargs)
    # --- Save sweep-level metadata (e.g., data subsampling seed) ---
    sweep_metadata = experiment.get_sweep_metadata(init_key)
    if sweep_metadata and not no_save:
        checkpoint_manager.save_sweep_metadata(sweep_metadata)

    # --- Subsample test set for faster evaluation if requested ---
    if max_eval_samples is not None and test_ds is not None:
        # Use a derived key for determinism, different from the training subsample key
        eval_subsample_key = jax.random.PRNGKey(init_key + 1)
        test_ds = _subsample_dataset(test_ds, max_eval_samples, eval_subsample_key)

    # Data loading failure is a fatal error for offline experiments, so abort the sweep.
    if train_ds is None and not experiment.is_online_experiment():
        logging.error("Failed to load dataset. Aborting sweep.")
        # Return copies to prevent external mutation of internal state
        return results_dict.copy(), failed_runs.copy()

    # 3. Run Sweep
    _execute_sweep_loops(
        experiment,
        batch_sizes,
        etas,
        init_key,
        no_save,
        eta_stability_search_depth,
        results_dict,
        failed_runs,
        checkpoint_manager,
        params0,
        model_for_runner,
        train_ds,
        test_ds,
        **kwargs,
    )

    # Return copies to prevent external mutation of internal state
    return results_dict.copy(), failed_runs.copy()


def get_trial_runner(context: TrialContext):
    """Factory function to create the appropriate trial runner."""
    try:
        runner_class = context.experiment.get_trial_runner_class()
        return runner_class(context=context)
    except NotImplementedError:
        logging.error(
            f"Experiment type {type(context.experiment).__name__} does not implement get_trial_runner_class()."
        )
        return None
