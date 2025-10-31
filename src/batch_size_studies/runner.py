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
from dataclasses import dataclass

import jax
import numpy as np
from tqdm.auto import tqdm

from .checkpoint_utils import CheckpointManager
from .definitions import RunKey
from .experiments import (
    LinearStudentExperiment,
    MLPStudentExperiment,
    MNIST1MExperiment,
    MNIST1MSampledExperiment,
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
    SyntheticExperimentLinearTeacher,
    SyntheticExperimentMLPTeacher,
)
from .paths import EXPERIMENTS_DIR


class CenteredModel:
    """
    A wrapper for a JAX model to compute centered outputs.

    This is used to match the loss function structure from training, which is
    L(p) = loss(model(p) - model(p0)), where p0 are the initial parameters.
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


def compute_model_widths(experiment) -> list[int]:
    """Computes the layer widths for the MLP model."""
    # MNIST experiments have a multi-class output, others have a single output.
    output_dim = getattr(experiment, "num_outputs", 1)
    return [experiment.D] + [experiment.N] * (experiment.L - 1) + [output_dim]


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


def _create_runner_kwargs(
    experiment,
    run_key: RunKey,
    params0,
    model_instance,
    checkpoint_manager: CheckpointManager,
    train_ds,
    test_ds,
    pbar,
    no_save: bool,
    init_key: int,
    num_steps: int,
    **kwargs,
) -> dict:
    """Assembles the keyword arguments for creating a TrialRunner."""
    base_kwargs = {
        "run_key": run_key,
        "params0": params0,
        "model_instance": model_instance,
        "checkpoint_manager": checkpoint_manager,
        "pbar": pbar,
        "no_save": no_save,
        "init_key": init_key,
        "num_steps": num_steps,
    }

    num_epochs = kwargs.get("num_epochs", getattr(experiment, "num_epochs", 1))

    if isinstance(experiment, (MNISTExperiment, MNIST1MExperiment, MNIST1MSampledExperiment)):
        base_kwargs.update({"num_epochs": num_epochs, "train_ds": train_ds, "test_ds": test_ds})
    elif isinstance(experiment, (SyntheticExperimentFixedData, SyntheticExperimentLinearTeacher)):
        base_kwargs.update({"num_epochs": num_epochs, "X_data": train_ds[0], "y_data": train_ds[1]})

    return base_kwargs


def _run_single_trial(
    experiment,
    run_key: RunKey,
    results_dict: dict,
    failed_runs: set,
    checkpoint_manager: CheckpointManager,
    params0,
    model_instance,
    train_ds,
    test_ds,
    pbar,
    no_save: bool,
    init_key: int,
    **kwargs,
) -> bool:
    """
    Checks the status of, runs, and validates a single trial configuration.
    Returns True if the run was successful (or already was), False otherwise.
    """
    num_steps = compute_num_steps(experiment, run_key.batch_size, train_ds, **kwargs)
    status = RunStatus(run_key, results_dict, failed_runs, num_steps, no_save)

    if not status.should_run:
        return status.is_successful

    runner_kwargs = _create_runner_kwargs(
        experiment,
        run_key,
        params0,
        model_instance,
        checkpoint_manager,
        train_ds,
        test_ds,
        pbar,
        no_save,
        init_key,
        num_steps,
        **kwargs,
    )

    trial_runner = _get_trial_runner(experiment, **runner_kwargs)

    if trial_runner:
        result = trial_runner.run()
        is_successful = validate_and_store_result(
            result, run_key, results_dict, failed_runs, experiment, checkpoint_manager, no_save
        )
    else:
        failed_runs.add(run_key)
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

    # The runner still needs to know the model's structure to initialize params
    if isinstance(experiment, LinearStudentExperiment):
        widths = [experiment.D, 1]
    elif isinstance(experiment, MLPStudentExperiment):
        widths = compute_model_widths(experiment)
    else:
        raise TypeError(f"Unknown student model for experiment type: {type(experiment).__name__}")

    params0 = initialize_model_params(model_instance, checkpoint_manager, init_key, widths, no_save)
    # Wrap the model for centering if it's an MLPStudentExperiment.
    # The trial runners will then use this centered model.
    if isinstance(experiment, MLPStudentExperiment):
        model_for_runner = CenteredModel(model_instance, params0)
    else:
        model_for_runner = model_instance

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
            is_successful = _run_single_trial(
                experiment=experiment,
                run_key=RunKey(batch_size=batch_size, eta=eta),
                results_dict=results_dict,
                failed_runs=failed_runs,
                checkpoint_manager=checkpoint_manager,
                params0=params0,
                model_instance=model_for_runner,
                train_ds=train_ds,
                test_ds=test_ds,
                pbar=eta_pbar,
                no_save=no_save,
                init_key=init_key,
                **kwargs,
            )

            if eta_tracker.update(is_successful):
                # Fast-forward the progress bar to the end for this batch size
                eta_pbar.update(len(sorted_etas) - eta_pbar.n)
                break

            eta_pbar.update(1)

        eta_pbar.close()
    return dict(results_dict), failed_runs


def _get_trial_runner(experiment, **runner_kwargs):
    """Factory function to create the appropriate trial runner."""
    try:
        runner_class = experiment.get_trial_runner_class()
        return runner_class(experiment=experiment, **runner_kwargs)
    except NotImplementedError:
        logging.error(f"Experiment type {type(experiment).__name__} does not implement get_trial_runner_class().")
        return None
