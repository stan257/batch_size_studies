"""
Unified Training Runner

This module provides a single, unified entry point for running all types of
hyperparameter sweeps (synthetic, MNIST, etc.). It centralizes the logic for
looping over hyperparameters, managing checkpoints, and saving results, while
dispatching to type-specific trial runners.
"""

import argparse
import logging
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import numpy as np
from tqdm.auto import tqdm

from .checkpoint_utils import CheckpointManager
from .configs import get_main_experiment_configs, get_main_hyperparameter_grids
from .constants import EVAL_SUBSAMPLE_SEED_OFFSET
from .definitions import LossType, OptimizerType, Parameterization, RunKey
from .experiments import MNIST1MExperiment
from .paths import EXPERIMENTS_DIR
from .protocols import ModelProtocol, TrainingOptions

# =============================================================================
# CLI Argument Helpers
# =============================================================================


def _coerce_enum(parser, enum_cls, raw_value, flag_name):
    if raw_value is None:
        return None
    candidate = raw_value.strip().lower()
    for member in enum_cls:
        if member.name.lower() == candidate or str(member.value).lower() == candidate:
            return member
    valid = ", ".join([f"{m.name} ({m.value})" for m in enum_cls])
    parser.error(f"Invalid value '{raw_value}' for {flag_name}. Valid choices: {valid}.")


def add_filter_args(parser: argparse.ArgumentParser):
    """Adds shared experiment filtering arguments to a parser."""
    parser.add_argument(
        "-n",
        "--name",
        nargs="*",
        help="Filter by the specific experiment name(s).",
    )
    parser.add_argument(
        "--experiment-type",
        action="append",
        dest="experiment_types",
        help="Filter experiments by their experiment_type string. Repeat the flag to include multiple types.",
    )
    parser.add_argument(
        "--optimizer",
        "--opt",
        dest="optimizer",
        help="Filter experiments by optimizer (e.g., SGD, Adam). Case-insensitive.",
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        help="Filter experiments by loss function (e.g., MSE, XENT). Case-insensitive.",
    )
    parser.add_argument(
        "--list-overrides",
        action="store_true",
        help="Show supported override keys (for -o KEY=VALUE) and exit.",
    )


def describe_supported_overrides() -> str:
    return "\n".join(
        [
            "Supported override keys:",
            "  - num_epochs=<int>: force a fixed number of epochs for fixed-data experiments.",
            "  - max_eval_samples=<int>: cap evaluation set size per epoch.",
            "  - save_interstitial_snapshots=<bool>: enable/disable dense weight snapshots.",
            "  - save_epoch_snapshots=<bool>: toggle per-epoch snapshots for fixed-data synthetic runs.",
            "  - disable_eval_dataset=<bool>: skip deterministic synthetic eval dataset (saves memory).",
            "  - dataset_loader=<callable>: alternate loader for MNIST-style datasets.",
            "  - forced_subsample_seed=<int>: deterministic seed for subsampled datasets.",
        ]
    )


def _resolve_experiment_configs(args: argparse.Namespace) -> dict[str, any] | None:
    """Builds the experiment dictionary based on CLI filter arguments."""
    optimizer_filter = _coerce_enum(argparse.ArgumentParser(), OptimizerType, args.optimizer, "--optimizer")
    loss_filter = _coerce_enum(argparse.ArgumentParser(), LossType, args.loss, "--loss")
    config_kwargs = {}
    if hasattr(args, "experiment_types") and args.experiment_types:
        config_kwargs["experiment_types"] = args.experiment_types
    if optimizer_filter is not None:
        config_kwargs["optimizer"] = optimizer_filter
    if loss_filter is not None:
        config_kwargs["loss_type"] = loss_filter
    experiments = get_main_experiment_configs(**config_kwargs)

    if args.name:
        experiments = {name: config for name, config in experiments.items() if name in args.name}
        if not experiments:
            logging.error(f"No experiments found with name(s): {args.name}. Aborting.")
            return None
    return experiments


def _handle_list_command(args: argparse.Namespace, experiments_to_run: dict[str, any]) -> None:
    """Handles the CLI 'list' command."""
    if getattr(args, "list_overrides", False):
        print(describe_supported_overrides())
        return

    logging.info("--- Available Experiments ---")
    headers = ["NAME", "TYPE", "OPTIMIZER", "LOSS"]
    rows = [
        [name, config.experiment_type, config.optimizer.name, config.loss_type.name]
        for name, config in experiments_to_run.items()
    ]

    if not rows:
        logging.info("No experiments match the provided filters.")
        return

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "=" * len(header_line)
    print(f"\n{separator}")
    print(f"Available Experiments ({len(rows)} total)")
    print(f"{separator}")
    print(header_line)
    print("-" * len(header_line))

    for row in sorted(rows):
        row_line = "  ".join(c.ljust(w) for c, w in zip(row, col_widths))
        print(row_line)

    print(f"{separator}\n")


def _apply_overrides(experiments_to_run: dict[str, any], overrides_list: list[str]) -> dict[str, any]:
    """Applies CLI overrides to experiment configs."""
    overrides = {}
    for override_str in overrides_list:
        key, value_str = override_str.split("=", 1)
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                if key == "parameterization":
                    value = Parameterization[value_str.upper()]
                else:
                    value = value_str
        overrides[key] = value

    logging.info(f"Applying overrides: {overrides}")
    return {name: replace(config, **overrides) for name, config in experiments_to_run.items()}


def _handle_run_command(args: argparse.Namespace, experiments_to_run: dict[str, any]) -> None:
    """Handles the CLI 'run' command."""
    directory = EXPERIMENTS_DIR
    batch_sizes, etas = get_main_hyperparameter_grids()
    if not experiments_to_run:
        logging.error("No experiments match the provided filters. Nothing to run.")
        return

    dry_run = getattr(args, "dry_run", False)
    dry_run_steps = getattr(args, "dry_run_steps", 5)

    if args.override:
        experiments_to_run = _apply_overrides(experiments_to_run, args.override)

    if dry_run:
        logging.info("Dry-run mode: selecting the first experiment and a single (B, η) pair.")
        first_item = next(iter(experiments_to_run.items()), None)
        if first_item is None:
            logging.error("No experiments available for dry-run.")
            return
        first_name, first_config = first_item
        dry_batch = min(batch_sizes) if batch_sizes else 1
        mid_eta = etas[len(etas) // 2] if etas else 0.1
        logging.info("Dry-run for '%s' @ (B=%s, η=%s) for %s steps.", first_name, dry_batch, mid_eta, dry_run_steps)
        run_experiment_sweep(
            experiment=first_config,
            batch_sizes=[dry_batch],
            etas=[mid_eta],
            init_key=0,
            directory=directory,
            dry_run=True,
            dry_run_steps=dry_run_steps,
            no_save=True,
            eta_stability_search_depth=None,
            max_eval_samples=args.max_eval_samples,
        )
        return

    filepaths = defaultdict(list)
    experiments_that_need_running = {}
    logging.info("--- Pre-flight check: Verifying experiments ---")

    for name, config in experiments_to_run.items():
        filepath = config.get_filepath(directory=directory)
        filepaths[filepath].append(name)

        if args.no_save:
            experiments_that_need_running[name] = config
        else:
            losses, failed = config.load_results(directory=directory, silent=True)
            if _all_runs_accounted_for(config, batch_sizes, etas, losses, failed):
                logging.info(f"  Skipping '{name}': Already complete. (Found file: {os.path.basename(filepath)})")
            else:
                logging.info(f"  Incomplete: '{name}'. Will run. (Checking file: {os.path.basename(filepath)})")
                experiments_that_need_running[name] = config

    has_collision = False
    for filepath, names in filepaths.items():
        if len(names) > 1:
            logging.error(f"Collision detected! Experiments {names} will write to the same file: {filepath}")
            has_collision = True

    if has_collision:
        logging.error("\nAborting due to filename collisions.")
        return

    if args.no_save:
        logging.info("\n--- --no-save enabled: All selected experiments will be run without saving. ---")

    if not experiments_that_need_running:
        logging.info("\n--- All experiments are already complete. Nothing to do. ---")
        return

    logging.info(f"\n--- Starting Pipeline for {len(experiments_that_need_running)} Incomplete Experiments ---")
    if args.num_processes <= 1:
        logging.info("Running experiments sequentially.")
        for name, config in experiments_that_need_running.items():
            try:
                _run_single_experiment(
                    name,
                    config,
                    batch_sizes,
                    etas,
                    directory,
                    args.no_save,
                    args.eta_stability_depth,
                    args.max_eval_samples,
                    args.save_interstitial_snapshots,
                    args.save_epoch_snapshots,
                )
            except Exception as exc:
                logging.error(f"Experiment '{name}' generated an exception: {exc}")
    else:
        logging.info(f"Running experiments with up to {args.num_processes} parallel workers.")
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            future_to_name = {
                executor.submit(
                    _run_single_experiment,
                    name,
                    config,
                    batch_sizes,
                    etas,
                    directory,
                    args.no_save,
                    args.eta_stability_depth,
                    args.max_eval_samples,
                    args.save_interstitial_snapshots,
                    args.save_epoch_snapshots,
                ): name
                for name, config in experiments_that_need_running.items()
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    future.result()
                except Exception as exc:
                    logging.error(f"Experiment '{name}' generated an exception: {exc}")

    logging.info("\n--- All experiments complete. ---")


def run_from_cli_args(args: argparse.Namespace):
    """
    Main orchestration logic driven by parsed command-line arguments.
    """
    experiments_to_run = _resolve_experiment_configs(args)
    if experiments_to_run is None:
        return

    match args.command:
        case "list":
            _handle_list_command(args, experiments_to_run)
        case "run":
            _handle_run_command(args, experiments_to_run)
        case _:
            logging.error(f"Unknown command: {args.command}")


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

    if isinstance(experiment_config, MNIST1MExperiment):
        from batch_size_studies.data_loading import load_mnist1m_dataset

        run_options["dataset_loader"] = load_mnist1m_dataset

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
