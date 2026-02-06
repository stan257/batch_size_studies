import logging
import os
import pickle

import jax
import numpy as np
from filelock import FileLock

from ..definitions import RunKey
from ..paths import EXPERIMENTS_DIR
from .storage_utils import CustomUnpickler, save_experiment

# =============================================================================
# Checkpoint and snapshot management
# =============================================================================


class CheckpointManager:
    """
    Manages saving and loading for both resumability (live checkpoints)
    and analysis (weight snapshots).
    """

    def __init__(self, experiment, directory=None):
        self.experiment = experiment
        base_dir = EXPERIMENTS_DIR if directory is None else os.fspath(directory)
        self.directory = os.path.abspath(base_dir)

        # Generate base path for this experiment
        self.exp_dir = os.path.join(self.directory, experiment.experiment_type)

        filename_variants = experiment.get_filename_variants(prefix="", extension="pkl")
        base_variants = [os.path.splitext(name)[0] for name in filename_variants]

        selected_base = None
        for base_candidate in base_variants:
            weights_path = os.path.join(self.exp_dir, f"{base_candidate}_weights.pkl")
            checkpoint_dir_candidate = os.path.join(self.exp_dir, f"{base_candidate}_checkpoints")
            if os.path.exists(weights_path) or os.path.exists(checkpoint_dir_candidate):
                selected_base = base_candidate
                break

        if selected_base is None:
            selected_base = base_variants[0]

        self._base_filename = selected_base

        # Path for live resume checkpoints
        self.checkpoint_dir = os.path.join(self.exp_dir, f"{selected_base}_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Path for analysis snapshots (weights)
        self.weights_filepath = os.path.join(self.exp_dir, f"{selected_base}_weights.pkl")

    def _get_resume_filepath(self, run_key: RunKey):
        run_key_str = f"B={run_key.batch_size}_eta={str(run_key.eta).replace('.', 'p')}"
        return os.path.join(self.checkpoint_dir, f"resume_{run_key_str}.pkl")

    def save_live_checkpoint(self, run_key: RunKey, step: int, params, opt_state, results: dict):
        """Saves the complete state needed to resume a trial using an atomic write."""
        filepath = self._get_resume_filepath(run_key)
        data = {
            "step": step,
            "params": params,
            "opt_state": opt_state,
            "results": results,
        }
        save_experiment(data, filepath)

    def load_live_checkpoint(self, run_key: RunKey):
        filepath = self._get_resume_filepath(run_key)
        if not os.path.exists(filepath):
            return None, None, {}, 0

        try:
            with open(filepath, "rb") as f:
                data = CustomUnpickler(f).load()

            # Backward compatibility logic
            if "results" in data:
                # New format: 'results' dict contains loss_history, etc.
                # 'step' is already a global step number.
                last_step = data.get("step", -1)
                results = data.get("results", {})
            else:
                # Old format: 'loss_history' is top-level, 'step' is an epoch number.
                logging.info(f"Detected old checkpoint format for {run_key}. Converting to new format.")
                last_epoch = data.get("step", -1)
                results = {"loss_history": data.get("loss_history", [])}

                # We need to determine steps_per_epoch to convert epoch to step.
                # This logic mirrors compute_num_steps from the runner.
                train_ds_size = getattr(self.experiment, "P", None)
                if train_ds_size is None:
                    # For MNIST, we need to estimate based on config.
                    train_ds_size = getattr(self.experiment, "max_train_samples", 60000)

                if train_ds_size is not None and run_key.batch_size > 0:
                    steps_per_epoch = train_ds_size // run_key.batch_size
                    last_step = (last_epoch + 1) * steps_per_epoch - 1
                else:
                    last_step = -1  # Fallback

            params = data.get("params")
            opt_state = data.get("opt_state")

            logging.info(f"Resuming run {run_key} from step {last_step + 1}")
            return params, opt_state, results, last_step + 1
        except (pickle.UnpicklingError, EOFError, AttributeError, KeyError) as e:
            logging.warning(f"Warning: Could not load checkpoint for {run_key}. Starting from scratch. Error: {e}")
            return None, None, {}, 0

    def save_analysis_snapshot(self, run_key: RunKey, step: int, params, initial_params):
        """
        Saves the delta of the weights for post-experiment analysis.
        """
        lock_path = self.weights_filepath + ".lock"
        with FileLock(lock_path):
            weights_data = {}
            if os.path.exists(self.weights_filepath):
                try:
                    with open(self.weights_filepath, "rb") as f:
                        weights_data = CustomUnpickler(f).load()
                except (pickle.UnpicklingError, EOFError):
                    logging.warning(
                        f"Could not load corrupted weights file {self.weights_filepath}. A new one will be created."
                    )
                    weights_data = {}

            if "initial_params" not in weights_data:
                weights_data["initial_params"] = initial_params
            if "weight_snapshots" not in weights_data:
                weights_data["weight_snapshots"] = {}

            delta_params = jax.tree_util.tree_map(lambda p, p0: p - p0, params, initial_params)

            # Manually handle nesting for regular dicts
            snapshots = weights_data["weight_snapshots"]
            if run_key not in snapshots:
                snapshots[run_key] = {}
            snapshots[run_key][step] = delta_params

            # Use atomic save_experiment to write back the updated data
            save_experiment(weights_data, self.weights_filepath)

    def load_initial_params(self):
        if not os.path.exists(self.weights_filepath):
            return None
        try:
            with open(self.weights_filepath, "rb") as f:
                data = CustomUnpickler(f).load()
            return data.get("initial_params")
        except (pickle.UnpicklingError, EOFError) as e:
            logging.warning(f"Could not load initial params from {self.weights_filepath}. Error: {e}")
            return None

    def initialize_and_save_initial_params(self, init_key: int, model_instance, widths: list[int]):
        """Generates and saves initial parameters if they don't exist, using a lock."""
        lock_path = self.weights_filepath + ".lock"
        with FileLock(lock_path):
            # Re-check if params were created by another process while waiting for the lock
            params0 = self.load_initial_params()
            if params0 is not None:
                return params0

            logging.info("No initial parameters found. Generating and saving.")
            params0 = model_instance.init_params(init_key, widths)
            weights_data = {"initial_params": params0, "weight_snapshots": {}}
            save_experiment(weights_data, self.weights_filepath)
            return params0

    def save_sweep_metadata(self, metadata_to_add: dict):
        """Saves or updates sweep-level metadata in the weights file."""
        lock_path = self.weights_filepath + ".lock"
        with FileLock(lock_path):
            weights_data = {}
            if os.path.exists(self.weights_filepath):
                try:
                    with open(self.weights_filepath, "rb") as f:
                        weights_data = CustomUnpickler(f).load()
                except (pickle.UnpicklingError, EOFError):
                    logging.warning(f"Corrupted weights file {self.weights_filepath}. Overwriting.")

            if "metadata" not in weights_data:
                weights_data["metadata"] = {}

            # Check for potential overwrites before updating to ensure data provenance.
            existing_metadata = weights_data["metadata"]
            for key, new_value in metadata_to_add.items():
                if key in existing_metadata and not np.array_equal(existing_metadata[key], new_value):
                    logging.warning(
                        f"Overwriting sweep metadata key '{key}'. "
                        f"Old value: {existing_metadata[key]}, New value: {new_value}. "
                        "This may invalidate provenance for existing results in this directory."
                    )

            weights_data["metadata"].update(metadata_to_add)
            save_experiment(weights_data, self.weights_filepath)

    def load_sweep_metadata(self) -> dict:
        """Loads sweep-level metadata from the weights file."""
        if not os.path.exists(self.weights_filepath):
            return {}
        try:
            with open(self.weights_filepath, "rb") as f:
                data = CustomUnpickler(f).load()
            return data.get("metadata", {})
        except (pickle.UnpicklingError, EOFError) as e:
            logging.warning(f"Could not load metadata from {self.weights_filepath}. Error: {e}")
            return {}

    def load_analysis_snapshot(self, run_key: RunKey, step: int):
        """
        Loads and reconstructs the full model parameters for a specific run and step
        from the analysis snapshot file.

        Returns:
            The reconstructed JAX PyTree of parameters, or None if not found.
        """
        if not os.path.exists(self.weights_filepath):
            logging.error(f"Weights file not found: {self.weights_filepath}")
            return None

        try:
            with open(self.weights_filepath, "rb") as f:
                data = CustomUnpickler(f).load()

            initial_params = data["initial_params"]
            delta_params = data["weight_snapshots"][run_key][step]

            return jax.tree_util.tree_map(lambda p0, d: p0 + d, initial_params, delta_params)
        except (KeyError, pickle.UnpicklingError, EOFError) as e:
            logging.error(f"Could not load snapshot for {run_key} at step {step}. Error: {e}")
            return None

    def load_full_weight_history(self, run_key: RunKey):
        """
        Loads and reconstructs the full history of model parameters for a specific run.

        This can be memory-intensive if many snapshots were saved.

        Args:
            run_key: The RunKey for the desired trial.

        Returns:
            A dictionary mapping step number to the reconstructed JAX PyTree of
            parameters for that step, or an empty dict if not found.
        """
        if not os.path.exists(self.weights_filepath):
            logging.info(f"Weights file not found: {self.weights_filepath}. Returning empty history.")
            return {}

        try:
            with open(self.weights_filepath, "rb") as f:
                data = CustomUnpickler(f).load()

            initial_params = data["initial_params"]
            run_snapshots = data["weight_snapshots"].get(run_key, {})

            return {
                step: jax.tree_util.tree_map(lambda p0, d: p0 + d, initial_params, delta_params)
                for step, delta_params in run_snapshots.items()
            }
        except (KeyError, pickle.UnpicklingError, EOFError) as e:
            logging.error(f"Could not load weight history for {run_key}. Error: {e}")
            return {}

    def load_final_params_for_all_runs(self) -> dict[RunKey, any]:
        """
        Loads the final parameters for every run stored in the weights file.
        """
        if not os.path.exists(self.weights_filepath):
            logging.info(f"Weights file not found: {self.weights_filepath}. Returning empty dict.")
            return {}

        try:
            with open(self.weights_filepath, "rb") as f:
                data = CustomUnpickler(f).load()
        except (pickle.UnpicklingError, EOFError) as e:
            logging.error(f"Could not load weights file {self.weights_filepath}: {e}")
            return {}

        initial_params = data.get("initial_params")
        if initial_params is None:
            logging.error(f"initial_params missing from weights file {self.weights_filepath}")
            return {}

        weight_snapshots: dict[RunKey, dict[int, any]] = data.get("weight_snapshots", {})
        final_params: dict[RunKey, any] = {}
        for run_key, step_map in weight_snapshots.items():
            if not step_map:
                continue
            final_step = max(step_map.keys())
            delta = step_map[final_step]
            final_params[run_key] = jax.tree_util.tree_map(lambda p0, d: p0 + d, initial_params, delta)
        return final_params

    def cleanup_live_checkpoint(self, run_key: RunKey):
        filepath = self._get_resume_filepath(run_key)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                logging.error(f"Error cleaning up checkpoint for {run_key}: {e}")


# =============================================================================
# Convenience loader helpers
# =============================================================================


def load_experiment_weights(
    experiment,
    batch_size: int,
    eta: float,
    directory: str | os.PathLike[str] | None = None,
    step_to_load: int | None = None,
):
    """
    A high-level utility to load saved model weights for a specific experiment run.

    If `step_to_load` is provided, it loads the weights for that specific step.
    If `step_to_load` is None, it loads the full history of all saved weights for the run.

    Args:
        experiment: The experiment configuration object.
        batch_size: The batch size of the desired run.
        eta: The learning rate (eta) of the desired run.
        directory: The base directory where experiments are saved.
        step_to_load: The specific step to load weights for. If None, the full
                      history is loaded.

    Returns:
        A JAX PyTree of parameters for a single step, a dictionary mapping
        step to parameters for the full history, or None/empty dict if not found.
    """
    manager = CheckpointManager(experiment, directory=directory)
    run_key = RunKey(batch_size=batch_size, eta=eta)

    if step_to_load is not None:
        return manager.load_analysis_snapshot(run_key, step_to_load)
    else:
        return manager.load_full_weight_history(run_key)


def load_final_weights_for_experiment(
    experiment,
    directory: str | os.PathLike[str] | None = None,
) -> dict[RunKey, any]:
    """
    Loads the final parameters for every completed run of the given experiment.
    """
    manager = CheckpointManager(experiment, directory=directory)
    return manager.load_final_params_for_all_runs()
