"""
Unified Training Runner

This module provides a single, unified entry point for running all types of
hyperparameter sweeps (synthetic, MNIST, etc.). It centralizes the logic for
looping over hyperparameters, managing checkpoints, and saving results, while
dispatching to type-specific trial runners.
"""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from tqdm.auto import tqdm

from .checkpoint_utils import CheckpointManager
from .definitions import RunKey
from .experiments import (
    MNIST1MExperiment,
    MNIST1MSampledExperiment,
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
    SyntheticExperimentLinearTeacher,
    SyntheticExperimentMLPTeacher,
)
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


class CenteredModel:
    """
    A wrapper for a JAX model to compute centered outputs. Does
        L(p) = loss(model(p) - model(p0)),
    where p0 are the initial parameters.
    """

    def __init__(self, model, params0):
        self.model = model
        self.params0 = params0
        # The model's __call__ needs to be jitted for performance.
        self.apply_fn = jax.jit(self.model)

    def __call__(self, params, inputs):
        """
        Computes model(params, inputs) - model(params0, inputs).
        """
        return self.apply_fn(params, inputs) - self.apply_fn(self.params0, inputs)


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
        loss_history = result.get("loss_history", [])
        return len(loss_history) >= self.num_steps

    @property
    def should_run(self) -> bool:
        """Determines if the trial should be executed."""
        if self.no_save:
            return True
        if self.run_key in self.failed_runs:
            logging.info(f"Skipping previously failed run {self.run_key}")
            return False
        if self.is_successful:
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
        results_dict, failed_runs = defaultdict(list), set()
    else:
        results_dict, failed_runs = experiment.load_results(directory=directory, silent=True)

    checkpoint_manager = CheckpointManager(experiment, directory=directory)
    return results_dict, failed_runs, checkpoint_manager


def initialize_model_params(
    model_instance, checkpoint_manager: CheckpointManager, init_key: int, widths: list[int], no_save: bool
):
    """Initializes or loads the initial model parameters (params0)."""
    if no_save:
        return model_instance.init_params(init_key, widths)

    # This method handles both loading and safe, locked initialization.
    return checkpoint_manager.initialize_and_save_initial_params(init_key, model_instance, widths)


# ============================================================================
# RUN CONFIGURATION HELPERS
# ============================================================================


def compute_num_steps(experiment, batch_size: int, train_ds, **kwargs) -> int:
    """Computes the total number of training steps for a trial."""
    if isinstance(experiment, (SyntheticExperimentFixedTime, SyntheticExperimentMLPTeacher)):
        return experiment.num_steps

    num_epochs = kwargs.get("num_epochs", getattr(experiment, "num_epochs", 1))

    if isinstance(experiment, (MNISTExperiment, MNIST1MExperiment, MNIST1MSampledExperiment)):
        num_train_samples = len(train_ds["image"])
    elif isinstance(experiment, (SyntheticExperimentFixedData, SyntheticExperimentLinearTeacher)):
        num_train_samples = experiment.P
    else:
        return 0

    steps_per_epoch = num_train_samples // batch_size
    return num_epochs * steps_per_epoch


# ============================================================================
# RESULT VALIDATION AND STORAGE
# ============================================================================


def validate_and_store_result(
    result: dict | None,
    run_key: RunKey,
    results_dict: dict,
    failed_runs: set,
    experiment,
    checkpoint_manager: CheckpointManager,
    no_save: bool,
) -> bool:
    """Validates the result of a trial and updates result/failure tracking."""
    # Remove any previous result for this key, successful or not
    results_dict.pop(run_key, None)
    failed_runs.discard(run_key)

    is_complete = experiment.is_run_complete(result, run_key) if result else False

    # A result is valid if it exists and doesn't contain non-finite values.
    is_valid = result is not None
    if "final_test_accuracy" in (result or {}) and not np.isfinite(result["final_test_accuracy"]):
        is_valid = False

    if is_valid:
        # Store any valid result, even if incomplete. It will be overwritten on resume.
        results_dict[run_key] = result
        if not no_save:
            if is_complete:
                checkpoint_manager.cleanup_live_checkpoint(run_key)
    else:
        # Only runs that are invalid (e.g., returned None from divergence) are marked as failures.
        # Incomplete runs are not failures.
        failed_runs.add(run_key)

    if not no_save:
        # Save results after every trial
        experiment.save_results(results_dict, failed_runs, os.path.dirname(checkpoint_manager.exp_dir))

    # The run is only considered "successful" for the purpose of the eta stability search
    # if it is both valid and fully complete.
    return is_valid and is_complete


# ============================================================================
# TRIAL EXECUTION HELPERS
# ============================================================================


def _run_single_trial(
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
        return status.is_successful

    trial_runner = _get_trial_runner(context)

    if trial_runner:
        result = trial_runner.run()
        is_successful = validate_and_store_result(
            result,
            context.run_key,
            results_dict,
            failed_runs,
            context.experiment,
            context.checkpoint_manager,
            context.no_save,
        )
    else:
        failed_runs.add(context.run_key)
        is_successful = False

    return is_successful


# ============================================================================
# MAIN SWEEP ORCHESTRATION
# ============================================================================


def run_experiment_sweep(
    experiment,
    batch_sizes: list[int],
    etas: list[float],
    init_key: int = 0,
    directory=EXPERIMENTS_DIR,
    no_save: bool = False,
    eta_stability_search_depth: int | None = None,
    **kwargs,
):
    """
    Orchestrates a full hyperparameter sweep, dispatching to the correct
    training logic based on the type of the experiment object.
    """
    # 1. Setup
    results_dict, failed_runs, checkpoint_manager = initialize_results_and_checkpoints(experiment, directory, no_save)

    # Polymorphic call to the experiment to create its own model instance.
    model_instance = experiment.create_model_instance()

    widths = experiment.get_model_widths()
    params0 = initialize_model_params(model_instance, checkpoint_manager, init_key, widths, no_save)
    model_for_runner = experiment.get_model_wrapper(model_instance, params0)  # e.g. centering the model or not

    # 2. Load Data
    train_ds, test_ds = experiment.prepare_datasets(init_key, **kwargs)
    if train_ds is None and not isinstance(experiment, (SyntheticExperimentFixedTime, SyntheticExperimentMLPTeacher)):
        logging.error("Failed to load dataset. Aborting sweep.")
        return dict(results_dict), failed_runs

    # 3. Run Sweep
    for batch_size in tqdm(batch_sizes, desc="Batch Size Sweep"):
        # Determine dataset size for the polymorphic check, if applicable.
        train_ds_size = None
        if isinstance(experiment, (MNISTExperiment, MNIST1MExperiment, MNIST1MSampledExperiment)) and train_ds:
            train_ds_size = len(train_ds["image"])

        # Polymorphic call to check if the batch size is valid.
        if experiment.should_skip_batch_size(batch_size, train_ds_size=train_ds_size):
            continue

        eta_tracker = EtaStabilityTracker(eta_stability_search_depth)

        sorted_etas = sorted(etas, reverse=True)
        eta_pbar = tqdm(total=len(sorted_etas), desc="Eta Sweep", leave=False)
        eta_pbar.reset()
        eta_pbar.set_description(f"Eta Sweep (B={batch_size})")

        for eta in sorted_etas:
            run_key = RunKey(batch_size=batch_size, eta=eta)
            # Pass num_epochs from kwargs if it exists, to correctly calculate total steps
            num_epochs_for_run = kwargs.get("num_epochs", getattr(experiment, "num_epochs", 1))
            num_steps = compute_num_steps(experiment, batch_size, train_ds, num_epochs=num_epochs_for_run)

            # Create the context object for this trial
            context = TrialContext(
                experiment=experiment,
                run_key=run_key,
                params0=params0,
                model_instance=model_for_runner,
                checkpoint_manager=checkpoint_manager,
                train_ds=train_ds,
                test_ds=test_ds,
                pbar=eta_pbar,
                no_save=no_save,
                init_key=init_key,
                num_steps=num_steps,
                num_epochs=num_epochs_for_run,
                kwargs=kwargs,
            )
            is_successful = _run_single_trial(context=context, results_dict=results_dict, failed_runs=failed_runs)

            if eta_tracker.update(is_successful):
                # Fast-forward the progress bar to the end for this batch size
                eta_pbar.update(len(sorted_etas) - eta_pbar.n)
                break

            eta_pbar.update(1)

        eta_pbar.close()
    return dict(results_dict), failed_runs


def _get_trial_runner(context: TrialContext):
    """Factory function to create the appropriate trial runner."""
    try:
        runner_class = context.experiment.get_trial_runner_class()
        return runner_class(context=context)
    except NotImplementedError:
        logging.error(
            f"Experiment type {type(context.experiment).__name__} does not implement get_trial_runner_class()."
        )
        return None


def are_all_runs_complete(
    experiment, losses: dict, failed_runs: set, batch_sizes: list[int], etas: list[float]
) -> bool:
    """
    Checks if all specified runs for an experiment are either completed, failed, or skipped.
    """
    for b in batch_sizes:
        # We pass train_ds_size=None because this is a pre-flight check
        # before data is loaded.
        if experiment.should_skip_batch_size(b, train_ds_size=None):
            continue

        for eta in etas:
            run_key = RunKey(b, eta)
            result = losses.get(run_key)
            is_failed = run_key in failed_runs

            if result is None and not is_failed:
                # The run is neither in the successful results nor in the failed set.
                # It's genuinely missing.
                return False

            if result is not None:
                # The run exists, check if it's complete.
                if not experiment.is_run_complete(result, run_key):
                    return False  # It's incomplete.

            # If result is None but is_failed is True, we consider it "accounted for"
            # and continue checking the next run.

    return True
