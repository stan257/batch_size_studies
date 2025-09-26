"""
Unified Training Runner

This module provides a single, unified entry point for running all types of
hyperparameter sweeps (synthetic, MNIST, etc.). It centralizes the logic for
looping over hyperparameters, managing checkpoints, and saving results, while
dispatching to type-specific trial runners.
"""

import logging
import pickle
from collections import defaultdict
from dataclasses import replace

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


def run_experiment_sweep(
    experiment,
    batch_sizes: list[int],
    etas: list[float],
    init_key: int = 0,
    directory=EXPERIMENTS_DIR,
    no_save: bool = False,
    **kwargs,
):
    """
    Orchestrates a full hyperparameter sweep, dispatching to the correct
    training logic based on the type of the experiment object.
    """
    # 1. Determine experiment family
    is_mnist = isinstance(experiment, (MNISTExperiment, MNIST1MExperiment, MNIST1MSampledExperiment))
    is_synthetic_fixed_data = isinstance(experiment, SyntheticExperimentFixedData)
    is_synthetic_fixed_time = isinstance(experiment, (SyntheticExperimentFixedTime, SyntheticExperimentMLPTeacher))

    # 2. Common setup for results and checkpointing
    if no_save:
        results_dict, failed_runs = defaultdict(list), set()
    else:
        results_dict, failed_runs = experiment.load_results(directory=directory)
    checkpoint_manager = CheckpointManager(experiment, directory=directory)
    mlp_instance = MLP(experiment.parameterization, experiment.gamma)

    # 3. Configure model output dimension
    output_dim = experiment.num_outputs if is_mnist else 1
    widths = [experiment.D] + [experiment.N] * (experiment.L - 1) + [output_dim]

    # 4. Setup initial parameters (params0)
    if no_save:
        params0 = mlp_instance.init_params(init_key, widths)
    else:
        params0 = checkpoint_manager.load_initial_params()
        if params0 is None:
            logging.info("No initial parameters found. Generating and saving them now.")
            params0 = mlp_instance.init_params(init_key, widths)
            weights_data = {"initial_params": params0, "weight_snapshots": {}}
            with open(checkpoint_manager.weights_filepath, "wb") as f:
                pickle.dump(weights_data, f)

    # 5. Load data based on experiment type
    train_ds, test_ds = None, None
    if is_mnist:
        dataset_loader = kwargs.get("dataset_loader")
        if dataset_loader is None:
            dataset_loader = load_datasets if isinstance(experiment, MNISTExperiment) else load_mnist1m_dataset

        try:
            (train_images, train_labels), (test_images, test_labels) = dataset_loader()

            if isinstance(experiment, MNIST1MSampledExperiment):
                num_samples_to_use = getattr(experiment, "max_train_samples", None)
                if num_samples_to_use is not None and num_samples_to_use > 0:
                    num_original_samples = len(train_images)
                    if num_original_samples >= num_samples_to_use:
                        shuffle_key = jr.PRNGKey(init_key)
                        indices_to_use = jr.permutation(shuffle_key, num_original_samples)[:num_samples_to_use]
                        train_images, train_labels = (
                            train_images[np.array(indices_to_use)],
                            train_labels[np.array(indices_to_use)],
                        )
                        logging.info(f"Training on a random subset of {len(train_images)} samples.")

            train_ds = {"image": train_images, "label": train_labels}
            test_ds = {"image": test_images, "label": test_labels}
        except FileNotFoundError as e:
            logging.error(f"Dataset not found: {e}")
            return dict(results_dict), failed_runs
    elif is_synthetic_fixed_data:
        data_key = jr.key(getattr(experiment, "seed", init_key))
        X_data, y_data = experiment.generate_data(data_key)
        train_ds = (X_data, y_data)

    # 6. Main sweep loop
    run_combinations = [(bs, e) for bs in batch_sizes for e in etas]
    pbar = tqdm(run_combinations, desc="Hyperparameter Sweep")

    for batch_size, eta in pbar:
        run_key = RunKey(batch_size=batch_size, eta=eta)
        pbar.set_description(f"Sweep (B={batch_size}, eta={eta:.2g})")

        # 7. Completion Check
        num_epochs = kwargs.get("num_epochs", getattr(experiment, "num_epochs", 1))
        if is_synthetic_fixed_time:
            num_steps = experiment.num_steps
        else:  # Epoch-based experiments
            num_train_samples = len(train_ds["image"]) if is_mnist else experiment.P
            if batch_size > num_train_samples:
                logging.warning(
                    f"Skipping run {run_key}: batch_size ({batch_size}) > dataset size ({num_train_samples})."
                )
                continue
            steps_per_epoch = num_train_samples // batch_size
            num_steps = num_epochs * steps_per_epoch

        if not no_save:
            if run_key in failed_runs or (
                run_key in results_dict and len(results_dict[run_key].get("loss_history", [])) >= num_steps
            ):
                logging.info(f"Skipping completed run {run_key}")
                continue

        # 8. Dispatch to the appropriate trial runner
        runner = None
        runner_kwargs = {
            "experiment": experiment,
            "run_key": run_key,
            "params0": params0,
            "mlp_instance": mlp_instance,
            "checkpoint_manager": checkpoint_manager,
            "pbar": pbar,
            "no_save": no_save,
            "init_key": init_key,
        }

        if is_mnist:
            current_experiment = experiment
            if "num_epochs" in kwargs and kwargs["num_epochs"] != experiment.num_epochs:
                current_experiment = replace(experiment, num_epochs=kwargs["num_epochs"])
            runner_kwargs["experiment"] = current_experiment
            runner_kwargs["train_ds"] = train_ds
            runner_kwargs["test_ds"] = test_ds
            runner = MNISTTrialRunner(**runner_kwargs)

        elif is_synthetic_fixed_data:
            runner_kwargs["num_epochs"] = num_epochs
            runner_kwargs["X_data"] = train_ds[0]
            runner_kwargs["y_data"] = train_ds[1]
            runner = SyntheticFixedDataTrialRunner(**runner_kwargs)

        elif is_synthetic_fixed_time:
            runner_kwargs["num_steps"] = num_steps
            runner = SyntheticFixedTimeTrialRunner(**runner_kwargs)

        if runner:
            result = runner.run()
        else:
            result = None

        # 9. Process and save results
        is_mnist_success = (
            is_mnist and result and "final_test_accuracy" in result and np.isfinite(result["final_test_accuracy"])
        )
        if result is None or (is_mnist and not is_mnist_success):
            failed_runs.add(run_key)
            if run_key in results_dict:
                del results_dict[run_key]
        else:
            results_dict[run_key] = result
            if not no_save:
                original_epochs = getattr(experiment, "num_epochs", 1)
                if is_mnist and len(result.get("epoch_test_accuracies", [])) >= original_epochs:
                    checkpoint_manager.cleanup_live_checkpoint(run_key)
                elif not is_mnist:
                    checkpoint_manager.cleanup_live_checkpoint(run_key)

        if not no_save:
            experiment.save_results(results_dict, failed_runs, directory)

    return dict(results_dict), failed_runs
