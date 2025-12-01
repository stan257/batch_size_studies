"""
Run Main Experiment Sweeps

This script orchestrates the main hyperparameter sweeps over batch size and
learning rate for all experiment configurations defined in `configs.py`.

It supports parallel execution, automatic completion checking, and dynamic
parameter overrides from the command line.
"""

import argparse
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime

from batch_size_studies.configs import get_main_experiment_configs, get_main_hyperparameter_grids
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization
from batch_size_studies.experiments import MNIST1MExperiment
from batch_size_studies.paths import EXPERIMENTS_DIR
from batch_size_studies.runner import _all_runs_accounted_for, run_experiment_sweep


def are_all_runs_accounted_for(config, losses: dict, failed: set, batch_sizes: list[int], etas: list[float]) -> bool:
    """Wrapper around the core runner pre-flight check for CLI use."""
    return _all_runs_accounted_for(config, batch_sizes, etas, losses, failed)


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"experiment_run_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


def run_single_experiment(
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
        logging.info(f"  Applying default num_epochs=1 for {type(experiment_config).__name__} experiment.")
        run_options["num_epochs"] = 1

    if isinstance(experiment_config, MNIST1MExperiment):
        from batch_size_studies.data_loading import load_mnist1m_dataset

        run_options["dataset_loader"] = load_mnist1m_dataset

    run_experiment_sweep(experiment=experiment_config, batch_sizes=batch_sizes, etas=etas, **run_options)

    logging.info(f"--- Finished Experiment: {name} ---")
    return name


def get_experiment_dir_by_name(name: str, base_dir: str = EXPERIMENTS_DIR) -> str | None:
    all_configs = get_main_experiment_configs()
    config = all_configs.get(name)
    if config is None:
        logging.error(f"Experiment with name '{name}' not found.")
        return None
    return os.path.join(base_dir, config.experiment_type)


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


def main():
    parser = argparse.ArgumentParser(description="Run and manage a series of ML experiments.")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- List Command ---

    list_parser = subparsers.add_parser("list", help="List all available experiments based on the provided filters.")

    add_filter_args(list_parser)

    # --- Run Command ---

    run_parser = subparsers.add_parser("run", help="Run a series of ML experiments.")

    add_filter_args(run_parser)

    run_parser.add_argument(
        "-o",
        "--override",
        action="append",  # Allows specifying multiple overrides
        help="Override a parameter for the selected runs, e.g., -o num_steps=100",
    )

    run_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run experiments without saving results to disk. Useful for validation and notebook runs.",
    )

    run_parser.add_argument(
        "--eta-stability-depth",
        type=int,
        default=None,
        help="Number of consecutive stable etas to find before stopping the sweep for a given batch size. If not set,"
        " all etas are run.",
    )

    run_parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Maximum number of test samples to use for evaluation each epoch. If not set, the full test set is used.",
    )

    run_parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of experiments to run in parallel (useful on clusters). Default=1 for sequential runs on a "
        "single machine.",
    )

    run_parser.add_argument(
        "--save-interstitial-snapshots",
        dest="save_interstitial_snapshots",
        action="store_true",
        help="Force saving interstitial weight snapshots between checkpoints.",
    )

    run_parser.add_argument(
        "--no-save-interstitial-snapshots",
        dest="save_interstitial_snapshots",
        action="store_false",
        help="Skip interstitial weight snapshots (fewer saved deltas, faster runs).",
    )

    run_parser.set_defaults(save_interstitial_snapshots=None)

    run_parser.add_argument(
        "--save-epoch-snapshots",
        dest="save_epoch_snapshots",
        action="store_true",
        help="Force saving weight snapshots at every epoch boundary for fixed-data synthetic runs.",
    )

    run_parser.add_argument(
        "--no-save-epoch-snapshots",
        dest="save_epoch_snapshots",
        action="store_false",
        help="Skip per-epoch snapshots for fixed-data synthetic runs to speed up long sweeps.",
    )

    run_parser.set_defaults(save_epoch_snapshots=None)

    args = parser.parse_args()

    setup_logging()

    # --- Build experiment list based on filters ---

    optimizer_filter = _coerce_enum(parser, OptimizerType, args.optimizer, "--optimizer")

    loss_filter = _coerce_enum(parser, LossType, args.loss, "--loss")

    config_kwargs = {}

    if args.experiment_types:
        config_kwargs["experiment_types"] = args.experiment_types

    if optimizer_filter is not None:
        config_kwargs["optimizer"] = optimizer_filter

    if loss_filter is not None:
        config_kwargs["loss_type"] = loss_filter

    experiments_to_run = get_main_experiment_configs(**config_kwargs)

    if args.name:
        experiments_to_run = {name: config for name, config in experiments_to_run.items() if name in args.name}

        if not experiments_to_run:
            logging.error(f"No experiments found with name(s): {args.name}. Aborting.")

            return

    # --- Command Dispatch ---

    if args.command == "list":
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

        return

    elif args.command == "run":
        directory = EXPERIMENTS_DIR

        batch_sizes, etas = get_main_hyperparameter_grids()

        if not experiments_to_run:
            logging.error("No experiments match the provided filters. Nothing to run.")

            return

        # Apply parameter overrides if provided

        if args.override:
            overrides = {}

            for override_str in args.override:
                key, value_str = override_str.split("=", 1)

                # Intelligently cast the value

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

            # Use dataclasses.replace to create new, modified experiment objects

            experiments_to_run = {name: replace(config, **overrides) for name, config in experiments_to_run.items()}

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

                if are_all_runs_accounted_for(config, losses, failed, batch_sizes, etas):
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
                    run_single_experiment(
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
                        run_single_experiment,
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


if __name__ == "__main__":
    main()
