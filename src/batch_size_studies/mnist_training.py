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
import tensorflow_datasets as tfds
from tqdm.auto import tqdm

from .checkpoint_utils import CheckpointManager
from .definitions import LossType, RunKey
from .experiments import MNIST1MExperiment, MNIST1MSampledExperiment, MNISTExperiment
from .models import MLP
from .paths import DATA_DIR, EXPERIMENTS_DIR
from .training_utils import create_optimizer


def _create_classification_loss_fn(apply_fn, experiment: MNISTExperiment | MNIST1MExperiment, params0):
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


def load_datasets():
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))

    # Return raw numpy arrays, conversion to JAX arrays will happen in the runner.
    train_images = train_ds["image"].astype(np.float32) / 255.0
    train_labels = train_ds["label"].astype(np.int32)
    test_images = test_ds["image"].astype(np.float32) / 255.0
    test_labels = test_ds["label"].astype(np.int32)

    return (train_images, train_labels), (test_images, test_labels)


def load_mnist1m_dataset(data_dir=DATA_DIR):
    """
    Loads the pre-processed MNIST-1M dataset from a local .npz file.
    """
    dataset_path = os.path.join(data_dir, "mnist1m", "mnist1m.npz")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"MNIST-1M dataset not found at '{dataset_path}'. "
            f"Please run `python scripts/process_mnist1m.py` script first."
        )

    with np.load(dataset_path) as data:
        X_train = data["X_train"].astype(np.float32) / 255.0
        y_train = data["y_train"].astype(np.int32)
        X_test = data["X_test"].astype(np.float32) / 255.0
        y_test = data["y_test"].astype(np.int32)

    # Add a channel dimension for compatibility with the training loop,
    # which expects a 4D tensor.
    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, axis=-1)
    if X_test.ndim == 3:
        X_test = np.expand_dims(X_test, axis=-1)

    return (X_train, y_train), (X_test, y_test)


def _run_single_trial_mnist(
    experiment: MNISTExperiment | MNIST1MExperiment,
    run_key: RunKey,
    params0,
    init_key: int,
    train_ds,
    test_ds,
    checkpoint_manager: CheckpointManager,
    mlp_instance: MLP,
    pbar=None,
    no_save: bool = False,
):
    """Executes a resumable training run for one set of hyperparameters."""
    batch_size, learning_rate = run_key.batch_size, run_key.eta

    # --- Load state from checkpoint or initialize ---
    if no_save:
        params, opt_state, results, start_epoch = None, None, {}, 0
    else:
        params, opt_state, results, start_epoch = checkpoint_manager.load_live_checkpoint(run_key)

    optimizer = create_optimizer(experiment, learning_rate)
    loss_fn = _create_classification_loss_fn(jax.jit(mlp_instance), experiment, params0)
    update_step = _create_update_step(loss_fn, optimizer)
    eval_step = _create_eval_step(jax.jit(mlp_instance), params0)

    if params is None:
        # First time running this trial, start from the common initial parameters.
        params = params0
        opt_state = optimizer.init(params)
        results = {"epoch_test_accuracies": [], "loss_history": []}
    else:
        if "loss_history" not in results:
            results["loss_history"] = []

    num_train = train_ds["image"].shape[0]
    num_steps_per_epoch = num_train // batch_size

    # --- Training Loop ---
    for epoch in range(start_epoch, experiment.num_epochs):
        rng = jr.PRNGKey(init_key + epoch + 1)  # Use a different key per epoch
        perms = jr.permutation(rng, num_train)
        perms = perms[: num_steps_per_epoch * batch_size]
        perms = perms.reshape((num_steps_per_epoch, batch_size))
        # Convert JAX permutations to NumPy to slice the host-resident NumPy arrays.
        np_perms = np.array(perms)

        pbar.set_description(
            f"Sweep (B={batch_size}, eta={learning_rate:.3g}) | Epoch {epoch + 1}/{experiment.num_epochs}"
        )

        # Collect JAX loss scalars for the epoch to avoid syncing every step
        epoch_losses = []
        for perm in np_perms:
            # Flatten images for the MLP
            batch_images = train_ds["image"][perm, ...].reshape(batch_size, -1)
            batch_labels = train_ds["label"][perm, ...]

            params, opt_state, loss, acc = update_step(params, opt_state, batch_images, batch_labels)
            epoch_losses.append(loss)

        # --- Post-Epoch Processing ---
        # Now, we block and get all the losses for the epoch at once, and check for divergence.
        epoch_losses_array = jnp.array(epoch_losses)
        if not jnp.all(jnp.isfinite(epoch_losses_array)):
            first_bad_loss = epoch_losses_array[~jnp.isfinite(epoch_losses_array)][0]
            logging.warning(f"Run {run_key} diverged with loss={first_bad_loss}. Stopping trial.")
            return None
        results["loss_history"].extend(epoch_losses_array.tolist())

        # --- Evaluation at end of epoch ---
        test_accuracies = []
        num_test = test_ds["image"].shape[0]
        eval_batch_size = 512
        test_steps = (num_test + eval_batch_size - 1) // eval_batch_size
        for i in range(test_steps):
            start_idx, end_idx = i * eval_batch_size, (i + 1) * eval_batch_size
            batch_images = test_ds["image"][start_idx:end_idx].reshape(-1, experiment.D)
            batch_labels = test_ds["label"][start_idx:end_idx]
            if batch_images.shape[0] > 0:
                acc = eval_step(params, batch_images, batch_labels)
                test_accuracies.append(acc)

        epoch_accuracy = float(jnp.mean(jnp.array(test_accuracies)))
        results["epoch_test_accuracies"].append(epoch_accuracy)
        pbar.set_postfix(accuracy=f"{epoch_accuracy:.4f}")

        # --- Save Checkpoint ---
        if not no_save:
            checkpoint_manager.save_live_checkpoint(run_key, epoch, params, opt_state, results)
            checkpoint_manager.save_analysis_snapshot(run_key, epoch, params, params0)

    # After all epochs, add the final accuracy to the results dict for convenience
    if results.get("epoch_test_accuracies"):
        results["final_test_accuracy"] = results["epoch_test_accuracies"][-1]

    return results


def run_mnist_experiment(
    experiment: MNISTExperiment | MNIST1MExperiment | MNIST1MSampledExperiment,
    batch_sizes: list[int],
    etas: list[float],
    dataset_loader: Callable | None = None,
    init_key: int = 0,
    directory=EXPERIMENTS_DIR,
    no_save: bool = False,
    **kwargs,
):
    """Orchestrates a hyperparameter sweep for MNIST experiments."""
    original_config_epochs = experiment.num_epochs

    if "num_epochs" in kwargs:
        runtime_epochs = kwargs.get("num_epochs")
        if runtime_epochs != experiment.num_epochs:
            logging.info(
                f"Overriding num_epochs from dataclass ({experiment.num_epochs}) with runtime value: {runtime_epochs}"
            )
            experiment = replace(experiment, num_epochs=runtime_epochs)

    if no_save:
        results_dict, failed_runs = defaultdict(list), set()
    else:
        results_dict, failed_runs = experiment.load_results(directory=directory)
    checkpoint_manager = CheckpointManager(experiment, directory=directory)

    # Create model instance and load/create initial parameters for analysis snapshots
    mlp_instance = MLP(experiment.parameterization, experiment.gamma)
    if no_save:
        widths = [experiment.D] + [experiment.N] * (experiment.L - 1) + [experiment.num_outputs]
        params0 = mlp_instance.init_params(init_key, widths)
    else:
        params0 = checkpoint_manager.load_initial_params()
        if params0 is None:
            logging.info("No initial parameters found for analysis snapshots. Generating and saving them now.")
            widths = [experiment.D] + [experiment.N] * (experiment.L - 1) + [experiment.num_outputs]
            params0 = mlp_instance.init_params(init_key, widths)
            weights_data = {"initial_params": params0, "weight_snapshots": {}}
            with open(checkpoint_manager.weights_filepath, "wb") as f:
                pickle.dump(weights_data, f)

    if dataset_loader is None:
        if isinstance(experiment, (MNIST1MExperiment, MNIST1MSampledExperiment)):
            dataset_loader = load_mnist1m_dataset
        else:
            # Default to the original loader for MNISTExperiment
            dataset_loader = load_datasets

    try:
        (train_images, train_labels), (test_images, test_labels) = dataset_loader()

        # Handle sampled datasets by taking a reproducible random subset
        if isinstance(experiment, MNIST1MSampledExperiment):
            num_samples_to_use = getattr(experiment, "max_train_samples", None)
            if num_samples_to_use is not None and num_samples_to_use > 0:
                num_original_samples = len(train_images)
                if num_original_samples < num_samples_to_use:
                    logging.warning(
                        f"Requested {num_samples_to_use} samples, but dataset only has "
                        f"{num_original_samples}. Using all available samples."
                    )
                else:
                    # Shuffle before slicing to get a random subset. Use init_key for reproducibility.
                    shuffle_key = jr.PRNGKey(init_key)
                    shuffled_indices = jr.permutation(shuffle_key, num_original_samples)

                    indices_to_use = shuffled_indices[:num_samples_to_use]
                    train_images = train_images[np.array(indices_to_use)]
                    train_labels = train_labels[np.array(indices_to_use)]
                    logging.info(
                        f"Training on a random subset of {len(train_images)} samples (from {num_original_samples})."
                    )

        # Keep datasets as NumPy arrays in host memory to avoid OOM on device.
        # Batches will be transferred to the device individually during training.
        train_ds = {"image": train_images, "label": train_labels}
        test_ds = {"image": test_images, "label": test_labels}
    except FileNotFoundError as e:
        logging.error(f"Dataset not found: {e}")
        return dict(results_dict), failed_runs

    run_combinations = [(bs, eta) for bs in batch_sizes for eta in etas]
    pbar = tqdm(run_combinations, desc="MNIST Sweep")

    for i, (batch_size, eta) in enumerate(pbar):
        run_key = RunKey(batch_size=batch_size, eta=eta)
        pbar.set_description(f"Sweep (B={batch_size}, eta={eta:.3g})")

        # Check if the run is already fully completed
        if not no_save:
            if (
                run_key in results_dict
                and len(results_dict[run_key].get("epoch_test_accuracies", [])) >= experiment.num_epochs
            ):
                logging.info(f"Skipping completed run {run_key}")
                continue
            if run_key in failed_runs:
                logging.info(f"Skipping failed run {run_key}")
                continue

        result = _run_single_trial_mnist(
            experiment=experiment,
            run_key=run_key,
            params0=params0,
            init_key=init_key,  # Use the same base init_key for all trials
            train_ds=train_ds,
            test_ds=test_ds,
            checkpoint_manager=checkpoint_manager,
            mlp_instance=mlp_instance,
            pbar=pbar,
            no_save=no_save,
        )

        if result is None or "final_test_accuracy" not in result or not jnp.isfinite(result["final_test_accuracy"]):
            failed_runs.add(run_key)
            if run_key in results_dict:
                del results_dict[run_key]
        else:
            results_dict[run_key] = result
            if not no_save:
                # Only clean up the checkpoint if the run is fully complete.
                if len(result.get("epoch_test_accuracies", [])) >= original_config_epochs:
                    checkpoint_manager.cleanup_live_checkpoint(run_key)

        if not no_save:
            experiment.save_results(results_dict, failed_runs, directory)

    return dict(results_dict), failed_runs
