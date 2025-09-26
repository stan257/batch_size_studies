"""
Run Small-muP Experiment Sweeps

This script is dedicated to running the "small-muP" experiments, which are
designed to find the smallest network widths at which mu-Parametrization
properties (like mu-transfer) become apparent.

It sweeps over the hyperparameter grids defined in `get_small_mup_...`
functions in `configs.py` and runs the corresponding experiments in parallel.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from batch_size_studies.configs import get_small_mup_experiment_configs, get_small_mup_hyperparameter_grids
from batch_size_studies.runner import run_experiment_sweep


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"small_mup_run_{timestamp}.log")

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


def run_single_experiment(name, experiment_config, batch_sizes, etas, directory="experiments", num_epochs=1):
    """
    A wrapper function to run a single experiment trial.
    This is designed to be called by the ProcessPoolExecutor.
    """
    logging.info(f"--- Starting Experiment: {name} ---")
    logging.info(f"Parameters: {experiment_config}")

    run_experiment_sweep(
        experiment=experiment_config,
        batch_sizes=batch_sizes,
        etas=etas,
        directory=directory,
        num_epochs=num_epochs,
    )
    logging.info(f"--- Finished Experiment: {name} ---")
    return name


def main():
    parser = argparse.ArgumentParser(description="Run a series of small muP experiments.")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of parallel processes. Defaults to number of CPUs.",
    )
    args = parser.parse_args()

    setup_logging()

    directory = "experiments"
    batch_sizes, etas = get_small_mup_hyperparameter_grids()
    num_epochs_for_mnist = 1

    experiments_to_run = get_small_mup_experiment_configs()

    logging.info("--- Pre-flight check: Verifying experiments ---")
    total_combinations = len(batch_sizes) * len(etas)
    experiments_that_need_running = {}

    for name, config in experiments_to_run.items():
        losses, failed = config.load_results(directory=directory)
        if len(losses) + len(failed) < total_combinations:
            experiments_that_need_running[name] = config
        else:
            logging.info(f"  Skipping '{name}': Already complete.")

    if not experiments_that_need_running:
        logging.info("\n--- All experiments are already complete. Nothing to do. ---")
        return

    logging.info(f"\n--- Starting Pipeline for {len(experiments_that_need_running)} Incomplete Experiments ---")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_name = {
            executor.submit(
                run_single_experiment,
                name,
                config,
                batch_sizes,
                etas,
                directory,
                num_epochs=num_epochs_for_mnist,
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
