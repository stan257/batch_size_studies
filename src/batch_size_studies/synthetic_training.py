import logging
import pickle
from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm.auto import tqdm

from .checkpoint_utils import CheckpointManager
from .definitions import RunKey
from .experiments import (
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
    SyntheticExperimentMLPTeacher,
)
from .models import MLP
from .training_utils import create_optimizer

# TODO: Figure out best place to keep track of implemented experiment types
EXPERIMENTS_TYPES = SyntheticExperimentFixedTime | SyntheticExperimentFixedData | SyntheticExperimentMLPTeacher


def _synthetic_loss_fn(params, x_batch, y_batch, mlp_instance, params0):
    """Calculates the regression loss against the initial model state."""
    pred = mlp_instance(params, x_batch) - mlp_instance(params0, x_batch)
    return jnp.mean((y_batch - pred) ** 2)


def _create_update_step(optimizer, mlp_instance, params0):
    """Creates a JIT-compiled function for a single training step."""
    # Use partial to bake in the static arguments for the loss function.
    # This makes the resulting function picklable for multiprocessing.
    partial_loss_fn = partial(_synthetic_loss_fn, mlp_instance=mlp_instance, params0=params0)

    @jax.jit
    def update_step(params, opt_state, x_batch, y_batch):
        loss, grad = jax.value_and_grad(partial_loss_fn)(params, x_batch, y_batch)
        updates, new_opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return update_step


def _get_snapshot_steps(max_steps: int) -> list[int]:
    """Generates a list of steps for checkpointing, e.g., [0, 1, 2, 5, 10, ...]."""
    steps = {0}
    for magnitude in [1, 10, 100, 1000, 10000, 100000, 1000000]:
        for base in [1, 2, 5]:
            step = base * magnitude
            if step < max_steps:
                steps.add(step)
    # Ensure the final state is always saved for analysis.
    if max_steps > 0:
        steps.add(max_steps - 1)
    return sorted(list(steps))


def _run_single_trial(
    experiment: EXPERIMENTS_TYPES,
    run_key: RunKey,
    params0,
    num_steps: int,
    checkpoint_manager: CheckpointManager,
    mlp_instance: MLP,
    pbar=None,
    no_save: bool = False,
):
    """Executes a generic training run for a single hyperparameter pair."""
    batch_size, eta = run_key.batch_size, run_key.eta

    # Load state from checkpoint if it exists; otherwise, initialize.
    if no_save:
        params, opt_state, saved_results, start_step = None, None, {}, 0
    else:
        params, opt_state, saved_results, start_step = checkpoint_manager.load_live_checkpoint(run_key)

    optimizer = create_optimizer(experiment, eta)
    update_step = _create_update_step(optimizer, mlp_instance, params0)

    if params is None:
        params = params0
        opt_state = optimizer.init(params)
        results = {"loss_history": [], "batch_key_seed": 0}
    else:
        results = saved_results

    # The seed for data generation must be part of the resumable state.
    batch_key_seed = results.get("batch_key_seed", 0)
    step_for_curr_data = 0  # This is always relative to the current X_data
    X_data, y_data = experiment.generate_data(jr.key(batch_key_seed))
    snapshot_steps = _get_snapshot_steps(num_steps)

    for current_step in range(start_step, num_steps):
        if (step_for_curr_data + 1) * batch_size > experiment.P:
            batch_key_seed += 1
            X_data, y_data = experiment.generate_data(jr.key(batch_key_seed))
            step_for_curr_data = 0

        start = step_for_curr_data * batch_size
        X_batch = X_data[start : start + batch_size]
        y_batch = y_data[start : start + batch_size]

        params, opt_state, loss = update_step(params, opt_state, X_batch, y_batch)
        step_for_curr_data += 1

        if not jnp.isfinite(loss):
            return None

        results["loss_history"].append(loss.item())
        results["batch_key_seed"] = batch_key_seed  # Keep the seed updated

        # Save checkpoint and analysis snapshot at prescribed steps
        if not no_save and current_step in snapshot_steps:
            checkpoint_manager.save_live_checkpoint(run_key, current_step, params, opt_state, results)
            checkpoint_manager.save_analysis_snapshot(run_key, current_step, params, params0)

        if pbar:
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=f"{current_step + 1}/{num_steps}")

    return results


def _run_single_trial_fixed_data(
    experiment: SyntheticExperimentFixedData,
    run_key: RunKey,
    params0,
    num_epochs: int,
    X_data: jnp.ndarray,
    y_data: jnp.ndarray,
    checkpoint_manager: CheckpointManager,
    mlp_instance: MLP,
    pbar=None,
    no_save: bool = False,
):
    """Executes a resumable training run for a fixed dataset over multiple epochs."""
    batch_size, eta = run_key.batch_size, run_key.eta

    # Load state from checkpoint or initialize.
    if no_save:
        params, opt_state, saved_results, start_step = None, None, {}, 0
    else:
        params, opt_state, saved_results, start_step = checkpoint_manager.load_live_checkpoint(run_key)

    optimizer = create_optimizer(experiment, eta)
    update_step = _create_update_step(optimizer, mlp_instance, params0)

    if params is None:
        params = params0
        opt_state = optimizer.init(params)
        results = {"loss_history": [], "epoch": 0}
    else:
        results = saved_results

    start_epoch: int = results.get("epoch", 0)
    num_train = X_data.shape[0]
    steps_per_epoch = num_train // batch_size
    num_steps = num_epochs * steps_per_epoch
    snapshot_steps = _get_snapshot_steps(num_steps)

    # Make sure we don't re-run from the start if we have a checkpoint
    current_step = start_step

    for epoch in range(start_epoch, num_epochs):
        # Use a different key per epoch for shuffling
        rng = jr.PRNGKey(getattr(experiment, "seed", 0) + epoch)
        perms = jr.permutation(rng, num_train)

        # Trim permutations to be a multiple of batch_size
        epoch_perms = perms[: steps_per_epoch * batch_size]
        epoch_perms = epoch_perms.reshape((steps_per_epoch, batch_size))

        pbar.set_description(f"Sweep (B={batch_size}, eta={eta:.2g}) | Epoch {epoch + 1}/{num_epochs}")

        for perm in epoch_perms:
            if current_step < start_step:
                current_step += 1
                continue

            X_batch, y_batch = X_data[perm, ...], y_data[perm, ...]
            params, opt_state, loss = update_step(params, opt_state, X_batch, y_batch)

            if not jnp.isfinite(loss):
                return None

            results["loss_history"].append(loss.item())
            results["epoch"] = epoch

            if not no_save and current_step in snapshot_steps:
                checkpoint_manager.save_live_checkpoint(run_key, current_step, params, opt_state, results)
                checkpoint_manager.save_analysis_snapshot(run_key, current_step, params, params0)

            if pbar:
                pbar.set_postfix(loss=f"{loss.item():.4f}", step=f"{current_step + 1}/{num_steps}")

            current_step += 1

    return results


def run_experiment(
    experiment: EXPERIMENTS_TYPES,
    batch_sizes: list[int],
    etas: list[float],
    init_key: int = 0,
    directory="experiments",
    no_save: bool = False,
    **kwargs,
) -> tuple[dict, set]:
    """
    Orchestrates a full hyperparameter sweep, dispatching to the correct
    training logic based on the type of the experiment object.
    """
    if not isinstance(
        experiment,
        (
            SyntheticExperimentFixedTime,
            SyntheticExperimentFixedData,
            SyntheticExperimentMLPTeacher,
        ),
    ):
        raise TypeError(f"Unsupported experiment type: {type(experiment).__name__}")

    # Setup managers for results and checkpoints
    if no_save:
        losses_dict, failed_runs = defaultdict(list), set()
    else:
        losses_dict, failed_runs = experiment.load_results(directory=directory)
    checkpoint_manager = CheckpointManager(experiment, directory=directory)

    # Define the model instance once, to be used for all operations.
    mlp_instance = MLP(experiment.parameterization, experiment.gamma)

    # Load or create initial parameters (params0)
    if no_save:
        # For no-save runs, just generate params0 in memory
        widths = [experiment.D] + [experiment.N] * (experiment.L - 1) + [1]
        params0 = mlp_instance.init_params(init_key, widths)
    else:
        params0 = checkpoint_manager.load_initial_params()
        if params0 is None:
            logging.info("No initial parameters found. Generating and saving them now.")
            widths = [experiment.D] + [experiment.N] * (experiment.L - 1) + [1]
            params0 = mlp_instance.init_params(init_key, widths)
            # Save immediately to the analysis file
            weights_data = {
                "initial_params": params0,
                "weight_snapshots": {},  # Use a regular dict to ensure it's picklable
            }
            with open(checkpoint_manager.weights_filepath, "wb") as f:
                pickle.dump(weights_data, f)

    # --- Dispatch based on experiment type ---
    is_fixed_data = isinstance(experiment, SyntheticExperimentFixedData)
    if is_fixed_data:
        logging.info(f"Generating fixed dataset for {experiment.experiment_type}...")
        # Use seed from experiment if available, otherwise use init_key
        data_key = jr.key(getattr(experiment, "seed", init_key))
        X_data, y_data = experiment.generate_data(data_key)

    run_combinations = [(bs, e) for bs in batch_sizes for e in etas]
    pbar = tqdm(run_combinations, desc="Hyperparameter Sweep")

    for batch_size, eta in pbar:
        run_key = RunKey(batch_size=batch_size, eta=eta)
        pbar.set_description(f"Sweep (B={batch_size}, eta={eta:.2g})")

        # Determine the number of steps based on the experiment type.
        if is_fixed_data:
            if batch_size > experiment.P:
                logging.warning(f"Skipping run {run_key}: batch_size ({batch_size}) > dataset size P ({experiment.P}).")
                continue
            num_epochs = kwargs.get("num_epochs", 1)
            steps_per_epoch = experiment.P // batch_size
            num_steps = num_epochs * steps_per_epoch
        else:  # FixedTime or MLPTeacher
            num_steps = experiment.num_steps

        if not no_save:
            if run_key in failed_runs or (
                run_key in losses_dict and len(losses_dict[run_key].get("loss_history", [])) >= num_steps
            ):
                logging.info(f"Skipping completed run {run_key}")
                continue

        # Call the appropriate trial runner
        if is_fixed_data:
            result = _run_single_trial_fixed_data(
                experiment=experiment,
                run_key=run_key,
                params0=params0,
                num_epochs=num_epochs,
                X_data=X_data,
                y_data=y_data,
                checkpoint_manager=checkpoint_manager,
                mlp_instance=mlp_instance,
                pbar=pbar,
                no_save=no_save,
            )
        else:
            result = _run_single_trial(
                experiment=experiment,
                run_key=run_key,
                params0=params0,
                num_steps=num_steps,
                checkpoint_manager=checkpoint_manager,
                mlp_instance=mlp_instance,
                pbar=pbar,
                no_save=no_save,
            )

        if result is None:
            failed_runs.add(run_key)
            if run_key in losses_dict:
                del losses_dict[run_key]
        else:
            # On successful completion, write final results to main file
            losses_dict[run_key] = result
            if not no_save:
                # And clean up the temporary resume checkpoint
                checkpoint_manager.cleanup_live_checkpoint(run_key)

        if not no_save:
            # Always save the main results file after each trial completes or fails
            experiment.save_results(losses_dict, failed_runs, directory)

    return dict(losses_dict), failed_runs
