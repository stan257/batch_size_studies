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

import jax.random as jr
import numpy as np
from tqdm.auto import tqdm

from .checkpoint_utils import CheckpointManager
from .data_loading import load_datasets, load_mnist1m_dataset
from .definitions import RunKey
from .experiments import (
    MNIST1MExperiment,
    MNIST1MSampledExperiment,
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
    SyntheticExperimentMLPTeacher,
)
from .models import MLP
from .paths import EXPERIMENTS_DIR
from .trainer import (
    MNISTTrialRunner,
    SyntheticFixedDataTrialRunner,
    SyntheticFixedTimeTrialRunner,
)


class ExperimentTypeChecker:
    """A helper class to determine the family of an experiment object."""

    def __init__(self, experiment):
        self.is_mnist = isinstance(experiment, (MNISTExperiment, MNIST1MExperiment, MNIST1MSampledExperiment))
        self.is_synthetic_fixed_data = isinstance(experiment, SyntheticExperimentFixedData)
        self.is_synthetic_fixed_time = isinstance(
            experiment, (SyntheticExperimentFixedTime, SyntheticExperimentMLPTeacher)
        )

    @property
    def uses_dataset(self) -> bool:
        return self.is_mnist or self.is_synthetic_fixed_data


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
    mlp_instance: MLP, checkpoint_manager: CheckpointManager, init_key: int, widths: list[int], no_save: bool
):
    """Initializes or loads the initial model parameters (params0)."""
    if no_save:
        return mlp_instance.init_params(init_key, widths)

    # This method handles both loading and safe, locked initialization.
    return checkpoint_manager.initialize_and_save_initial_params(init_key, mlp_instance, widths)


# ============================================================================
# RUN CONFIGURATION HELPERS
# ============================================================================


def compute_model_widths(experiment, type_checker: ExperimentTypeChecker) -> list[int]:
    """Computes the layer widths for the MLP model."""
    output_dim = experiment.num_outputs if type_checker.is_mnist else 1
    return [experiment.D] + [experiment.N] * (experiment.L - 1) + [output_dim]


def compute_num_steps(experiment, type_checker: ExperimentTypeChecker, batch_size: int, train_ds, **kwargs) -> int:
    """Computes the total number of training steps for a trial."""
    if type_checker.is_synthetic_fixed_time:
        return experiment.num_steps

    num_epochs = kwargs.get("num_epochs", getattr(experiment, "num_epochs", 1))

    if type_checker.is_mnist:
        num_train_samples = len(train_ds["image"])
    elif type_checker.is_synthetic_fixed_data:
        num_train_samples = experiment.P
    else:
        return 0

    steps_per_epoch = num_train_samples // batch_size
    return num_epochs * steps_per_epoch


def should_skip_batch_size(batch_size: int, train_ds, type_checker: ExperimentTypeChecker, experiment) -> bool:
    """Checks if a batch size is valid for the given experiment and dataset."""
    if type_checker.is_synthetic_fixed_time:
        return False

    num_train_samples = len(train_ds["image"]) if type_checker.is_mnist else experiment.P
    if batch_size > num_train_samples:
        logging.warning(
            f"Skipping run configurations for batch_size ({batch_size}) > dataset size ({num_train_samples})."
        )
        return True
    return False


# ============================================================================
# RESULT VALIDATION AND STORAGE
# ============================================================================


def validate_and_store_result(
    result: dict | None,
    run_key: RunKey,
    type_checker: ExperimentTypeChecker,
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

    is_mnist_success = (
        type_checker.is_mnist
        and result
        and "final_test_accuracy" in result
        and np.isfinite(result["final_test_accuracy"])
    )
    is_synthetic_success = not type_checker.is_mnist and result is not None

    is_successful = is_mnist_success or is_synthetic_success

    if is_successful:
        results_dict[run_key] = result
        if not no_save:
            # Only cleanup if the run is fully complete.
            is_fully_complete = False
            if type_checker.is_mnist:
                # The `experiment` object here is the original one, so it has the correct total number of epochs.
                original_epochs = getattr(experiment, "num_epochs", 1)
                if len(result.get("epoch_test_accuracies", [])) >= original_epochs:
                    is_fully_complete = True
            else:  # For synthetic, we assume any success is a full run for now.
                is_fully_complete = True

            if is_fully_complete:
                checkpoint_manager.cleanup_live_checkpoint(run_key)
    else:
        failed_runs.add(run_key)

    if not no_save:
        # Save results after every trial
        experiment.save_results(results_dict, failed_runs, os.path.dirname(checkpoint_manager.exp_dir))

    return is_successful


# ============================================================================
# TRIAL EXECUTION HELPERS
# ============================================================================


def _create_runner_kwargs(
    experiment,
    run_key: RunKey,
    type_checker: ExperimentTypeChecker,
    params0,
    mlp_instance: MLP,
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
        "experiment": experiment,
        "run_key": run_key,
        "params0": params0,
        "mlp_instance": mlp_instance,
        "checkpoint_manager": checkpoint_manager,
        "pbar": pbar,
        "no_save": no_save,
        "init_key": init_key,
        "num_steps": num_steps,
    }

    num_epochs = kwargs.get("num_epochs", getattr(experiment, "num_epochs", 1))

    if type_checker.is_mnist:
        base_kwargs.update({"num_epochs": num_epochs, "train_ds": train_ds, "test_ds": test_ds})
    elif type_checker.is_synthetic_fixed_data:
        base_kwargs.update({"num_epochs": num_epochs, "X_data": train_ds[0], "y_data": train_ds[1]})

    return base_kwargs


def _run_single_trial(
    experiment,
    run_key: RunKey,
    type_checker: ExperimentTypeChecker,
    results_dict: dict,
    failed_runs: set,
    checkpoint_manager: CheckpointManager,
    params0,
    mlp_instance: MLP,
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
    num_steps = compute_num_steps(experiment, type_checker, run_key.batch_size, train_ds, **kwargs)
    status = RunStatus(run_key, results_dict, failed_runs, num_steps, no_save)

    if not status.should_run:
        return status.is_successful

    runner_kwargs = _create_runner_kwargs(
        experiment,
        run_key,
        type_checker,
        params0,
        mlp_instance,
        checkpoint_manager,
        train_ds,
        test_ds,
        pbar,
        no_save,
        init_key,
        num_steps,
        **kwargs,
    )

    trial_runner = _get_trial_runner(type_checker, **runner_kwargs)

    if trial_runner:
        result = trial_runner.run()
        is_successful = validate_and_store_result(
            result, run_key, type_checker, results_dict, failed_runs, experiment, checkpoint_manager, no_save
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
    type_checker = ExperimentTypeChecker(experiment)
    results_dict, failed_runs, checkpoint_manager = initialize_results_and_checkpoints(experiment, directory, no_save)
    mlp_instance = MLP(experiment.parameterization, experiment.gamma)
    widths = compute_model_widths(experiment, type_checker)
    params0 = initialize_model_params(mlp_instance, checkpoint_manager, init_key, widths, no_save)

    # 2. Load Data
    train_ds, test_ds = prepare_datasets(experiment, type_checker, init_key, **kwargs)
    if type_checker.uses_dataset and train_ds is None:
        logging.error("Failed to load dataset. Aborting sweep.")
        return dict(results_dict), failed_runs

    # 3. Run Sweep
    sorted_etas = sorted(etas, reverse=True)
    eta_pbar = tqdm(total=len(sorted_etas), desc="Eta Sweep", leave=False)
    for batch_size in tqdm(batch_sizes, desc="Batch Size Sweep"):
        if should_skip_batch_size(batch_size, train_ds, type_checker, experiment):
            continue

        eta_tracker = EtaStabilityTracker(eta_stability_search_depth)

        eta_pbar.reset()
        eta_pbar.set_description(f"Eta Sweep (B={batch_size})")

        for eta in sorted_etas:
            is_successful = _run_single_trial(
                experiment=experiment,
                run_key=RunKey(batch_size=batch_size, eta=eta),
                type_checker=type_checker,
                results_dict=results_dict,
                failed_runs=failed_runs,
                checkpoint_manager=checkpoint_manager,
                params0=params0,
                mlp_instance=mlp_instance,
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


# ============================================================================
# TRIAL RUNNER AND DATASET HELPERS
# ============================================================================


def _get_trial_runner(type_checker: ExperimentTypeChecker, **runner_kwargs):
    """Factory function to create the appropriate trial runner."""
    if type_checker.is_mnist:
        return MNISTTrialRunner(**runner_kwargs)
    elif type_checker.is_synthetic_fixed_data:
        return SyntheticFixedDataTrialRunner(**runner_kwargs)
    elif type_checker.is_synthetic_fixed_time:
        return SyntheticFixedTimeTrialRunner(**runner_kwargs)
    else:
        logging.error(f"Unknown experiment type for experiment: {runner_kwargs['experiment'].experiment_type}")
        return None


def _subsample_mnist_data(train_images, train_labels, experiment, init_key):
    """Helper to subsample MNIST training data."""
    num_samples_to_use = getattr(experiment, "max_train_samples", None)
    if num_samples_to_use is not None and num_samples_to_use > 0:
        num_original_samples = len(train_images)
        if num_original_samples >= num_samples_to_use:
            shuffle_key = jr.PRNGKey(init_key)
            indices_to_use = jr.permutation(shuffle_key, num_original_samples)[:num_samples_to_use]
            train_images = train_images[np.array(indices_to_use)]
            train_labels = train_labels[np.array(indices_to_use)]
            logging.info(f"Training on a random subset of {len(train_images)} samples.")
    return train_images, train_labels


def _load_mnist_dataset(experiment, init_key: int, dataset_loader=None):
    """Loads MNIST dataset with optional subsampling."""
    if dataset_loader is None:
        dataset_loader = load_datasets if isinstance(experiment, MNISTExperiment) else load_mnist1m_dataset

    try:
        (train_images, train_labels), (test_images, test_labels) = dataset_loader()

        if isinstance(experiment, MNIST1MSampledExperiment):
            train_images, train_labels = _subsample_mnist_data(train_images, train_labels, experiment, init_key)

        train_ds = {"image": train_images, "label": train_labels}
        test_ds = {"image": test_images, "label": test_labels}
        return train_ds, test_ds

    except FileNotFoundError as e:
        logging.error(f"Dataset not found: {e}")
        return None, None


def prepare_datasets(experiment, type_checker: ExperimentTypeChecker, init_key: int, **kwargs):
    """Prepares training and test datasets based on the experiment type."""
    train_ds, test_ds = None, None
    if type_checker.is_mnist:
        return _load_mnist_dataset(experiment, init_key, kwargs.get("dataset_loader"))

    elif type_checker.is_synthetic_fixed_data:
        data_key = jr.key(getattr(experiment, "seed", init_key))
        X_data, y_data = experiment.generate_data(data_key)
        train_ds = (X_data, y_data)
        test_ds = None

    return train_ds, test_ds
