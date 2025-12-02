"""
Run Main Experiment Sweeps

This script provides the primary command-line interface for running and managing
hyperparameter sweeps defined in the `batch_size_studies` library. It acts as
a thin wrapper around the core orchestration logic in `runner.py`.
"""

import argparse
import logging
import os
from datetime import datetime

from batch_size_studies.runner import add_filter_args, run_from_cli_args


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


def main():
    """Parses CLI arguments and dispatches to the core runner logic."""
    parser = argparse.ArgumentParser(description="Run a series of ML experiments.")
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

    run_from_cli_args(args)


if __name__ == "__main__":
    main()
