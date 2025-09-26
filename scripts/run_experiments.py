"""
Run Main Experiment Sweeps

This script orchestrates the main hyperparameter sweeps over batch size and
learning rate for all experiment configurations defined in `configs.py`.

It supports parallel execution, automatic completion checking, and dynamic
parameter overrides from the command line.
"""

import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime

from batch_size_studies.configs import get_main_experiment_configs, get_main_hyperparameter_grids
from batch_size_studies.definitions import Parameterization
from batch_size_studies.experiments import (
    MNIST1MExperiment,
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
    SyntheticExperimentMLPTeacher,
)
from batch_size_studies.mnist_training import load_mnist1m_dataset, run_mnist_experiment
from batch_size_studies.paths import EXPERIMENTS_DIR
from batch_size_studies.synthetic_training import run_experiment as run_synthetic_experiment


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
    num_epochs_for_fixed_data=1,
    directory=EXPERIMENTS_DIR,
    no_save: bool = False,
):
    """
    A wrapper function to run a single experiment trial. This is designed
    to be called by the ProcessPoolExecutor.
    """
    logging.info(f"--- Starting Experiment: {name} ---")
    if no_save:
        logging.warning(f"Running in no-save mode for {name}. Results will NOT be saved.")
    logging.info(f"Parameters: {experiment_config}")

    run_options = {"num_epochs": num_epochs_for_fixed_data, "directory": directory, "no_save": no_save}

    match experiment_config:
        case SyntheticExperimentFixedTime() | SyntheticExperimentFixedData() | SyntheticExperimentMLPTeacher():
            run_synthetic_experiment(
                experiment=experiment_config,
                batch_sizes=batch_sizes,
                etas=etas,
                **run_options,
            )
        case MNISTExperiment():
            run_mnist_experiment(
                experiment=experiment_config,
                batch_sizes=batch_sizes,
                etas=etas,
                **run_options,
            )
        case MNIST1MExperiment():
            run_mnist_experiment(
                experiment=experiment_config,
                batch_sizes=batch_sizes,
                etas=etas,
                dataset_loader=load_mnist1m_dataset,
                **run_options,
            )
        case _:
            logging.error(f"Unknown experiment type for '{name}': {type(experiment_config).__name__}")

    logging.info(f"--- Finished Experiment: {name} ---")
    return name


def get_experiment_dir_by_name(name: str, base_dir: str = EXPERIMENTS_DIR) -> str | None:
    all_configs = get_main_experiment_configs()
    config = all_configs.get(name)
    if config is None:
        logging.error(f"Experiment with name '{name}' not found.")
        return None
    return os.path.join(base_dir, config.experiment_type)


def main():
    parser = argparse.ArgumentParser(description="Run a series of ML experiments.")
    parser.add_argument(
        "-n",
        "--name",
        nargs="*",  # Allows specifying zero, one, or multiple names
        help="Run only the experiment(s) with these specific names. "
        "If not provided, all defined experiments will be run.",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",  # Allows specifying multiple overrides
        help="Override a parameter for the selected runs, e.g., -o num_steps=100",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run experiments without saving results to disk. Useful for validation and notebook runs.",
    )
    args = parser.parse_args()

    setup_logging()

    directory = EXPERIMENTS_DIR
    num_epochs_for_fixed_data = 1
    batch_sizes, etas = get_main_hyperparameter_grids()
    experiments_to_run = get_main_experiment_configs()

    # --- Apply CLI Arguments ---
    # 1. Filter experiments by name if provided
    if args.name:
        experiments_to_run = {name: config for name, config in experiments_to_run.items() if name in args.name}
        if not experiments_to_run:
            logging.error(f"No experiments found with name(s): {args.name}. Aborting.")
            return

    # 2. Apply parameter overrides if provided
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

    # --- Pre-flight Safety and Completion Checks ---
    filepaths = defaultdict(list)
    experiments_that_need_running = {}
    logging.info("--- Pre-flight check: Verifying experiments ---")
    total_combinations = len(batch_sizes) * len(etas)

    for name, config in experiments_to_run.items():
        filepath = config.get_filepath(directory=directory)
        filepaths[filepath].append(name)

        if args.no_save:
            # If not saving, we intend to run everything regardless of completion status
            experiments_that_need_running[name] = config
        else:
            losses, failed = config.load_results(directory=directory)
            if len(losses) + len(failed) < total_combinations:
                experiments_that_need_running[name] = config
            else:
                logging.info(f"  Skipping '{name}': Already complete.")

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

    # --- Main Experiment Loop (Parallel Execution) ---
    logging.info(f"\n--- Starting Pipeline for {len(experiments_that_need_running)} Incomplete Experiments ---")
    with ProcessPoolExecutor() as executor:
        future_to_name = {
            executor.submit(
                run_single_experiment,
                name,
                config,
                batch_sizes,
                etas,
                num_epochs_for_fixed_data,
                directory,
                args.no_save,
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
