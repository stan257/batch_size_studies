import logging
import os
import pickle
from collections import defaultdict
from dataclasses import replace
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from tqdm.auto import tqdm

from .checkpoint_utils import CheckpointManager
from .definitions import LossType, Parameterization, RunKey
from .experiments import MNIST1MExperiment, MNIST1MSampledExperiment, MNISTExperiment
from .mnist_training import (
    load_datasets,
    load_mnist1m_dataset,
)
from .models import MLP
from .paths import EXPERIMENTS_DIR


def _create_classification_loss_fn(apply_fn, experiment: MNISTExperiment | MNIST1MSampledExperiment, params0):
    match experiment.loss_type:
        case LossType.XENT:

            def loss_fn(params, x_batch, y_batch_labels):
                logits = apply_fn(params, x_batch) - apply_fn(params0, x_batch)
                one_hot_labels = jax.nn.one_hot(y_batch_labels, num_classes=experiment.num_outputs)
                loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels))
                return loss, logits

            return loss_fn
        case LossType.MSE:

            def loss_fn(params, x_batch, y_batch_labels):
                logits = apply_fn(params, x_batch) - apply_fn(params0, x_batch)
                one_hot_labels = jax.nn.one_hot(y_batch_labels, num_classes=experiment.num_outputs)
                # Use element-wise mean squared error. TODO: Figure out how this scales with num_outputs
                loss = jnp.mean((logits - one_hot_labels) ** 2)
                return loss, logits

            return loss_fn
        case _:
            raise NotImplementedError(f"Loss type {experiment.loss_type} not implemented for classification.")


def _create_update_step(loss_fn, optimizer):
    @jax.jit
    def update_step(params, opt_state, x_batch, y_batch):
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_batch, y_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == y_batch)
        return new_params, new_opt_state, loss, accuracy

    return update_step


def _create_eval_step(apply_fn, params0):
    @jax.jit
    def eval_step(params, x_batch, y_batch):
        # Evaluation should also be on the centered function
        logits = apply_fn(params, x_batch) - apply_fn(params0, x_batch)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == y_batch)
        return accuracy

    return eval_step


def run_mnist_gamma_eta_sweep(
    base_experiment: MNISTExperiment | MNIST1MExperiment | MNIST1MSampledExperiment,
    batch_size: int,
    gamma_range: int,
    gamma_res: int,
    eta_range: int,
    eta_res: int,
    logspace_eta_range: int = 3,
    dataset_loader: Callable | None = None,
    init_key: int = 0,
    directory: str = EXPERIMENTS_DIR,
    no_save: bool = False,
    save_subfolder: str | None = "sanity_checks",
    **kwargs,
):
    """
    Orchestrates a hyperparameter sweep for MNIST experiments, fixing batch size
    and sweeping over gamma and eta.

    This routine is designed for exploring the interaction between the MLP's
    gamma parameter and the learning rate, for a fixed batch size.

    For each `gamma`, it creates a separate experiment configuration, allowing
    for independent checkpointing and results storage. This ensures that runs
    with different `gamma` values do not interfere with each other.

    Args:
        base_experiment: The base experiment configuration. Its `gamma` value
                         will be overridden by the values in `gammas`.
        batch_size: The fixed batch size for all training runs.
        gamma_range: The log10 range for the gamma sweep (e.g., 2 for 10^-2 to 10^2).
        gamma_res: The resolution (points per decade) for the gamma sweep.
        eta_range: The log10 range for the eta sweep.
        eta_res: The resolution for the eta sweep.
        logspace_eta_range: Once a stable LR is found, how many decades below it
                            to continue searching. Defaults to 3.
        dataset_loader: A function to load the dataset. If None, it defaults
                        to the appropriate MNIST or MNIST-1M loader.
        init_key: The base random key for parameter initialization.
        directory: The root directory to save experiment results and checkpoints.
        save_subfolder: If provided, results are saved in a subfolder of `directory`.
                        Defaults to "sanity_checks". Set to None to save in `directory`.
        no_save: If True, do not save any checkpoints or results to disk. This
                 is useful for debugging or quick tests.
        **kwargs: Additional arguments that can override experiment settings at
                  runtime, e.g., `num_epochs`.

    Returns:
        A tuple containing:
        - A dictionary of results structured as `{gamma: {RunKey(...): result_dict}}`.
        - A dictionary of failed runs structured as `{gamma: {RunKey(...): RunKey(...)}}`.
    """
    if not no_save and save_subfolder:
        directory = os.path.join(directory, save_subfolder)
        os.makedirs(directory, exist_ok=True)

    # Generate hyperparameter grids
    gammas = np.logspace(-gamma_range, gamma_range, 1 + 2 * gamma_range * gamma_res)
    etas = np.logspace(eta_range, -eta_range, 1 + 2 * eta_range * eta_res)  # Descending

    # --- 1a. Create Initial Parameters (gamma-independent) ---
    # Since params0 are gamma-independent, we generate them once before the sweep.
    mlp_for_init = MLP(base_experiment.parameterization, gamma=1.0)  # Gamma doesn't affect init
    widths = [base_experiment.D] + [base_experiment.N] * (base_experiment.L - 1) + [base_experiment.num_outputs]
    if no_save:
        params0 = mlp_for_init.init_params(init_key, widths)
    else:
        # Use a manager for the base experiment to handle the shared params0
        base_checkpoint_manager = CheckpointManager(base_experiment, directory=directory)
        params0 = base_checkpoint_manager.load_initial_params()
        if params0 is None:
            logging.info("No shared initial params found. Generating and saving.")
            params0 = mlp_for_init.init_params(init_key, widths)
            weights_data = {"initial_params": params0, "weight_snapshots": {}}
            with open(base_checkpoint_manager.weights_filepath, "wb") as f:
                pickle.dump(weights_data, f)

    # These will store the aggregated results from all gamma-specific files.
    all_results = defaultdict(dict)
    all_failed_runs = defaultdict(dict)

    # --- 1. Load Dataset ---
    # Load dataset once to avoid repeated I/O during the sweep.
    if dataset_loader is None:
        if isinstance(base_experiment, MNIST1MExperiment | MNIST1MSampledExperiment):
            dataset_loader = load_mnist1m_dataset
        else:
            dataset_loader = load_datasets

    try:
        (train_images, train_labels), (test_images, test_labels) = dataset_loader()
        # Permute the entire dataset once before any slicing.
        num_original_samples = len(train_images)
        shuffle_key = jr.PRNGKey(init_key)
        shuffled_indices = jr.permutation(shuffle_key, num_original_samples)
        train_images = train_images[shuffled_indices]
        train_labels = train_labels[shuffled_indices]
        logging.info(f"Shuffled training set of {num_original_samples} samples using key {init_key}.")

        # If the experiment specifies a subset of training samples, slice the pre-shuffled dataset.
        num_samples_to_use = getattr(base_experiment, "max_train_samples", None)
        if num_samples_to_use is not None and num_samples_to_use > 0:
            if num_original_samples < num_samples_to_use:
                logging.warning(
                    f"Requested {num_samples_to_use} samples, but dataset only has "
                    f"{num_original_samples}. Using all available samples."
                )
            train_images = train_images[:num_samples_to_use]
            train_labels = train_labels[:num_samples_to_use]
            logging.info(f"Training on {len(train_images)} samples (sliced from pre-shuffled data).")

        train_ds = {"image": train_images, "label": train_labels}
        test_ds = {"image": test_images, "label": test_labels}
    except FileNotFoundError as e:
        logging.error(f"Dataset not found: {e}")
        return dict(all_results), dict(all_failed_runs)

    # --- 2. Main Sweep Loop ---
    for gamma in tqdm(gammas, desc="Gamma Sweep"):
        found_first_stable_eta = False
        log_eta_lower_bound = -eta_range

        # Create experiment config, model, and checkpoint manager for this gamma
        experiment = replace(base_experiment, gamma=gamma)
        mlp_instance = MLP(experiment.parameterization, experiment.gamma)
        checkpoint_manager = CheckpointManager(experiment, directory=directory)

        # Load results for this gamma ONCE before the eta loop to avoid repeated I/O
        # and confusing log messages.
        if not no_save:
            results_dict, failed_runs = experiment.load_results(directory=directory)
        else:
            results_dict, failed_runs = defaultdict(dict), set()

        for eta in tqdm(etas, desc=f"Eta Sweep (γ={gamma:.2f})", leave=False):
            if np.log10(eta) < log_eta_lower_bound:
                break

            run_key = RunKey(batch_size=batch_size, eta=eta)
            original_config_epochs = experiment.num_epochs
            if "num_epochs" in kwargs:
                runtime_epochs = kwargs.get("num_epochs")
                if runtime_epochs != experiment.num_epochs:
                    experiment = replace(experiment, num_epochs=runtime_epochs)

            # Check for existing results
            if not no_save:
                if (
                    run_key in results_dict
                    and len(results_dict[run_key].get("epoch_test_accuracies", [])) >= experiment.num_epochs
                ):
                    logging.info(f"Skipping completed run {run_key} for γ={gamma}")
                    all_results[gamma][eta] = results_dict[run_key]
                    if not found_first_stable_eta:
                        found_first_stable_eta = True
                        log_eta_lower_bound = np.log10(eta) - logspace_eta_range
                    continue
                if run_key in failed_runs:
                    logging.info(f"Skipping failed run {run_key} for γ={gamma}")
                    continue

            # --- Inlined Training Trial ---
            if no_save:
                params, opt_state, trial_results, start_epoch = None, None, {}, 0
            else:
                params, opt_state, trial_results, start_epoch = checkpoint_manager.load_live_checkpoint(run_key)
            # NB: we do not adjust eta in terms of gamma by design -- this differs from the main mnist_training routine!
            lr = eta * experiment.N if experiment.parameterization == Parameterization.MUP else eta
            optimizer = optax.sgd(learning_rate=lr)
            loss_fn = _create_classification_loss_fn(jax.jit(mlp_instance), experiment, params0)
            update_step = _create_update_step(loss_fn, optimizer)
            eval_step = _create_eval_step(jax.jit(mlp_instance), params0)

            if params is None:
                params = params0
                opt_state = optimizer.init(params)
                trial_results = {"epoch_test_accuracies": [], "loss_history": []}
            if "loss_history" not in trial_results:
                trial_results["loss_history"] = []

            num_train = train_ds["image"].shape[0]
            num_steps_per_epoch = num_train // batch_size
            diverged = False

            for epoch in range(start_epoch, experiment.num_epochs):
                epoch_losses = []
                for t in range(num_steps_per_epoch):
                    start_idx, end_idx = t * batch_size, (t + 1) * batch_size
                    batch_images = train_ds["image"][start_idx:end_idx].reshape(batch_size, -1)
                    batch_labels = train_ds["label"][start_idx:end_idx]
                    params, opt_state, loss, acc = update_step(params, opt_state, batch_images, batch_labels)
                    epoch_losses.append(loss)

                epoch_losses_array = jnp.array(epoch_losses)
                if not jnp.all(jnp.isfinite(epoch_losses_array)):
                    first_bad_loss = epoch_losses_array[~jnp.isfinite(epoch_losses_array)][0]
                    logging.warning(f"Run {run_key} for γ={gamma} diverged with loss={first_bad_loss}.")
                    diverged = True
                    break

                trial_results["loss_history"].extend(epoch_losses_array.tolist())

                test_accuracies = []
                num_test, eval_batch_size = test_ds["image"].shape[0], 512
                test_steps = (num_test + eval_batch_size - 1) // eval_batch_size
                for i in range(test_steps):
                    start_idx, end_idx = i * eval_batch_size, (i + 1) * eval_batch_size
                    batch_images = test_ds["image"][start_idx:end_idx].reshape(-1, experiment.D)
                    batch_labels = test_ds["label"][start_idx:end_idx]
                    if batch_images.shape[0] > 0:
                        test_accuracies.append(eval_step(params, batch_images, batch_labels))

                epoch_accuracy = float(jnp.mean(jnp.array(test_accuracies)))
                trial_results["epoch_test_accuracies"].append(epoch_accuracy)

                if not no_save:
                    checkpoint_manager.save_live_checkpoint(run_key, epoch, params, opt_state, trial_results)
                    checkpoint_manager.save_analysis_snapshot(run_key, epoch, params, params0)

            # --- Process Results & Stability Logic ---
            if diverged:
                all_failed_runs[gamma][eta] = run_key
                failed_runs.add(run_key)
                if run_key in results_dict:
                    del results_dict[run_key]
            else:  # Converged
                if trial_results.get("epoch_test_accuracies"):
                    trial_results["final_test_accuracy"] = trial_results["epoch_test_accuracies"][-1]
                all_results[gamma][eta] = trial_results
                results_dict[run_key] = trial_results
                failed_runs.discard(run_key)

                if len(trial_results.get("epoch_test_accuracies", [])) >= original_config_epochs:
                    checkpoint_manager.cleanup_live_checkpoint(run_key)

                if not found_first_stable_eta:
                    found_first_stable_eta = True
                    log_eta_lower_bound = np.log10(eta) - logspace_eta_range

            if not no_save:
                experiment.save_results(results_dict, failed_runs, directory)

    return dict(all_results), dict(all_failed_runs)


def load_mnist_gamma_eta_sweep_results(
    base_experiment: MNISTExperiment | MNIST1MExperiment | MNIST1MSampledExperiment,
    gamma_range: int,
    gamma_res: int,
    directory: str = EXPERIMENTS_DIR,
    save_subfolder: str | None = "sanity_checks",
):
    """
    Loads and aggregates results from a gamma-eta sweep.

    This function is the counterpart to `run_mnist_gamma_eta_sweep`. It knows
    how to find and parse the results that are stored in separate files for
    each gamma value within a given subfolder.

    Args:
        base_experiment: The base experiment configuration used for the sweep.
        gamma_range: The log10 range for the gamma sweep.
        gamma_res: The resolution for the gamma sweep.
        directory: The root directory where experiments were saved.
        save_subfolder: The subfolder used during the sweep (e.g., "sanity_checks").

    Returns:
        A tuple containing:
        - A dictionary of results structured as `{gamma: {RunKey(...): result_dict}}`.
        - A dictionary of failed runs structured as `{gamma: {RunKey(...): RunKey(...)}}`.
    """
    if save_subfolder:
        directory = os.path.join(directory, save_subfolder)

    gammas = np.logspace(-gamma_range, gamma_range, 1 + 2 * gamma_range * gamma_res)

    all_results = defaultdict(dict)
    all_failed_runs = defaultdict(dict)

    logging.info(f"Loading sweep results from: {directory}")

    for gamma in gammas:
        # Recreate the specific experiment config to find the correct results file
        experiment = replace(base_experiment, gamma=gamma)
        try:
            results_for_gamma, failures_for_gamma = experiment.load_results(directory=directory)
            all_results[gamma] = dict(results_for_gamma)
            all_failed_runs[gamma] = {rk: rk for rk in failures_for_gamma}
        except FileNotFoundError:
            logging.warning(f"Results file for gamma={gamma} not found. Skipping.")

    return dict(all_results), dict(all_failed_runs)
