"""
Unified Training Runner

This module provides a single, unified entry point for running all types of
hyperparameter sweeps (synthetic, MNIST, etc.). It centralizes the logic for
looping over hyperparameters, managing checkpoints, and saving results, while
dispatching to type-specific trial runners.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from tqdm.auto import tqdm

from ..constants import EVAL_SUBSAMPLE_SEED_OFFSET
from ..definitions import RunKey
from ..paths import EXPERIMENTS_DIR
from .checkpoint_utils import CheckpointManager
from .protocols import ModelProtocol, TrainingOptions

# =============================================================================
# Core Runner Logic
# =============================================================================


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


def _run_single_experiment(
    name,
    experiment_config,
    batch_sizes,
    etas,
    directory=EXPERIMENTS_DIR,
    no_save: bool = False,
    eta_stability_search_depth: int | None = None,
    max_eval_samples: int | None = None,
    save_interstitial_snapshots: bool | None = None,
    save_epoch_snapshots: bool | None = None,
):
    """
    A wrapper function to run a single experiment trial. This is designed
    to be called by the ProcessPoolExecutor.
    """
    logging.info(f"--- Starting Experiment: {name} ---")
    if no_save:
        logging.warning(f"Running in no-save mode for {name}. Results will NOT be saved.")
    logging.info(f"Parameters: {experiment_config}")

    run_options = {
        "directory": directory,
        "no_save": no_save,
        "eta_stability_search_depth": eta_stability_search_depth,
        "max_eval_samples": max_eval_samples,
    }
    if save_interstitial_snapshots is not None:
        run_options["save_interstitial_snapshots"] = save_interstitial_snapshots
    if save_epoch_snapshots is not None:
        run_options["save_epoch_snapshots"] = save_epoch_snapshots

    # Selectively apply a default number of epochs only if the experiment
    # configuration does not already specify one.
    if not hasattr(experiment_config, "num_epochs"):
        # Some online experiments (fixed-time synthetic sweeps, etc.) train for a fixed step budget
        # instead of epochs. Those dataclasses omit `num_epochs`, so we fall back to 1 epoch for
        # compatibility. If you add a new offline experiment, declare num_epochs explicitly so this
        # branch never fires by accident.
        logging.info(f"  Applying default num_epochs=1 for {type(experiment_config).__name__} experiment.")
        run_options["num_epochs"] = 1

    default_loader = getattr(experiment_config, "get_default_dataset_loader", lambda: None)()
    if default_loader is not None and "dataset_loader" not in run_options:
        run_options["dataset_loader"] = default_loader

    run_experiment_sweep(experiment=experiment_config, batch_sizes=batch_sizes, etas=etas, **run_options)

    logging.info(f"--- Finished Experiment: {name} ---")
    return name


@dataclass
class TrialContext:
    """
    A dependency-injection container for a single experimental trial.

    This object bundles all the necessary state and configuration (the experiment
    spec, the specific hyperparameter key, model instances, data, etc.) needed
    to execute one run. It is created by the main sweep orchestrator and passed
    to the `TrialRunner` subclass.
    """

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
    options: TrainingOptions = field(default_factory=TrainingOptions)
    kwargs: dict = field(default_factory=dict)
    # Data fields, can be None
    train_ds: Any | None = None
    test_ds: Any | None = None


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
        expected_steps = result.get("expected_steps")
        expected_epochs = result.get("expected_epochs")
        loss_history = result.get("loss_history", [])
        epoch_accs = result.get("epoch_test_accuracies", [])

        if expected_steps is not None:
            return len(loss_history) >= expected_steps
        if expected_epochs is not None:
            return len(epoch_accs) >= expected_epochs

        # Fallback to current sweep request if no explicit expectations are stored.
        return len(loss_history) >= self.num_steps

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


def _is_run_result_complete(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False

    if "expected_steps" in result:
        expected_steps = result.get("expected_steps")
        if expected_steps is not None:
            return len(result.get("loss_history", [])) >= expected_steps
    elif "expected_epochs" in result:
        expected_epochs = result.get("expected_epochs")
        if expected_epochs is not None:
            return len(result.get("epoch_test_accuracies", [])) >= expected_epochs

    epoch_accs = result.get("epoch_test_accuracies")
    if isinstance(epoch_accs, list) and len(epoch_accs) > 0:
        return True

    loss_history = result.get("loss_history")
    if isinstance(loss_history, list) and len(loss_history) > 0:
        return True

    return False


def _all_runs_accounted_for(experiment, batch_sizes, etas, results_dict, failed_runs) -> bool:
    """Fast pre-flight check to determine if the sweep already has complete results."""
    for batch_size in batch_sizes:
        if experiment.should_skip_batch_size(batch_size, train_ds=None):
            continue
        for eta in etas:
            run_key = RunKey(batch_size=batch_size, eta=eta)
            if run_key in failed_runs:
                continue
            if not _is_run_result_complete(results_dict.get(run_key)):
                return False
    return True


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


def _get_git_revision() -> str:
    """Return the current git commit hash (or 'UNKNOWN' if not available)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def _build_training_options_from_kwargs(options_kwargs: dict) -> TrainingOptions:
    max_eval_samples = options_kwargs.get("max_eval_samples")
    save_interstitial = options_kwargs.get("save_interstitial_snapshots", False)
    epoch_snapshots = options_kwargs.get("save_epoch_snapshots", True)
    disable_eval_dataset = options_kwargs.get("disable_eval_dataset", False)
    return TrainingOptions(
        max_eval_samples=max_eval_samples,
        save_interstitial_snapshots=bool(save_interstitial),
        save_epoch_snapshots=bool(epoch_snapshots),
        disable_eval_dataset=bool(disable_eval_dataset),
    )


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
    training_options = _build_training_options_from_kwargs(kwargs)
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
            steps_to_run = kwargs.get("dry_run_steps") if kwargs.get("dry_run") else num_steps
            with tqdm(total=steps_to_run, desc=f"η={eta:.3g}", leave=False) as pbar:
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
                    num_steps=steps_to_run,
                    num_epochs=num_epochs_for_run,
                    options=training_options,
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
    save_interstitial_snapshots: bool = False,
    save_epoch_snapshots: bool | None = None,
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

    if not no_save and _all_runs_accounted_for(experiment, batch_sizes, etas, results_dict, failed_runs):
        logging.info("All requested (B, η) combinations already complete. Skipping sweep.")
        return results_dict.copy(), failed_runs.copy()

    # 2. Load Data
    train_ds, test_ds = experiment.prepare_datasets(init_key, **kwargs)
    # Save sweep-level metadata (e.g., data subsampling seed)
    sweep_metadata = {"init_key": init_key}
    sweep_metadata.update(experiment.get_sweep_metadata(init_key))
    if not no_save:
        checkpoint_manager.save_sweep_metadata(sweep_metadata)

    # Subsample test set for faster evaluation if requested
    if max_eval_samples is not None and test_ds is not None:
        # Use a derived key for determinism, different from the training subsample key
        eval_subsample_key = jax.random.PRNGKey(init_key + EVAL_SUBSAMPLE_SEED_OFFSET)
        test_ds = _subsample_dataset(test_ds, max_eval_samples, eval_subsample_key)

    # Data loading failure is a fatal error for offline experiments, so abort the sweep.
    if train_ds is None and not experiment.is_online_experiment():
        logging.error("Failed to load dataset. Aborting sweep.")
        # Return copies to prevent external mutation of internal state
        return results_dict.copy(), failed_runs.copy()

    # 3. Run Sweep
    kwargs.setdefault("save_interstitial_snapshots", save_interstitial_snapshots)
    if max_eval_samples is not None:
        kwargs.setdefault("max_eval_samples", max_eval_samples)
    if not kwargs.get("save_interstitial_snapshots", False):
        logging.warning(
            "*** WARNING: save_interstitial_snapshots is OFF; only initial weights will remain in _weights.pkl. "
            "Enable --save-interstitial-snapshots if you need intermediate models for analysis. ***"
        )
    if save_epoch_snapshots is not None:
        kwargs.setdefault("save_epoch_snapshots", save_epoch_snapshots)

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
    if not no_save:
        # Record the exact source revision so every sweep is traceable to a commit.
        git_hash = _get_git_revision()
        metadata = {"git_commit": git_hash}
        checkpoint_manager.save_sweep_metadata(metadata)
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
