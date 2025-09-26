import logging
import os
import pickle

import jax

from .definitions import RunKey
from .storage_utils import CustomUnpickler, generate_experiment_filename


class CheckpointManager:
    """
    Manages saving and loading for both resumability (live checkpoints)
    and analysis (weight snapshots).
    """

    def __init__(self, experiment, directory="experiments"):
        self.experiment = experiment

        # Generate base path for this experiment
        exp_params = experiment.to_params_dict()
        full_filename = generate_experiment_filename(exp_params, prefix="", extension="pkl")
        base_filename = os.path.splitext(full_filename)[0]
        self.exp_dir = os.path.join(directory, experiment.experiment_type)

        # Path for live resume checkpoints
        self.checkpoint_dir = os.path.join(self.exp_dir, f"{base_filename}_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Path for analysis snapshots (weights)
        self.weights_filepath = os.path.join(self.exp_dir, f"{base_filename}_weights.pkl")

    def _get_resume_filepath(self, run_key: RunKey):
        """Generates a unique filepath for a trial's live checkpoint."""
        run_key_str = f"B={run_key.batch_size}_eta={str(run_key.eta).replace('.', 'p')}"
        return os.path.join(self.checkpoint_dir, f"resume_{run_key_str}.pkl")

    def save_live_checkpoint(self, run_key: RunKey, step: int, params, opt_state, results: dict):
        """Saves the complete state needed to resume a trial."""
        filepath = self._get_resume_filepath(run_key)
        data = {
            "step": step,
            "params": params,
            "opt_state": opt_state,
            "results": results,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_live_checkpoint(self, run_key: RunKey):
        """Loads the state for a trial. Returns defaults if no checkpoint exists."""
        filepath = self._get_resume_filepath(run_key)
        if not os.path.exists(filepath):
            return None, None, {}, 0

        try:  # Catch specific, expected errors during file loading
            with open(filepath, "rb") as f:
                data = CustomUnpickler(f).load()

            # Default to -1 so start_step is 0
            last_step = data.get("step", -1)
            params = data.get("params")
            opt_state = data.get("opt_state")
            results = data.get("results", {})

            logging.info(f"Resuming run {run_key} from step {last_step + 1}")
            return params, opt_state, results, last_step + 1
        except (pickle.UnpicklingError, EOFError, AttributeError, KeyError) as e:
            logging.warning(f"Warning: Could not load checkpoint for {run_key}. Starting from scratch. Error: {e}")
            return None, None, {}, 0

    def save_analysis_snapshot(self, run_key: RunKey, step: int, params, initial_params):
        """Saves the delta of the weights for post-experiment analysis."""
        # Load existing weights data
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

        # Initialize structure if not present
        if "initial_params" not in weights_data:
            weights_data["initial_params"] = initial_params
        if "weight_snapshots" not in weights_data:
            weights_data["weight_snapshots"] = {}

        # Compute and store delta
        delta_params = jax.tree_util.tree_map(lambda p, p0: p - p0, params, initial_params)

        # Manually handle nesting for regular dicts
        snapshots = weights_data["weight_snapshots"]
        if run_key not in snapshots:
            snapshots[run_key] = {}
        snapshots[run_key][step] = delta_params

        # Save back
        with open(self.weights_filepath, "wb") as f:
            pickle.dump(weights_data, f)

    def load_initial_params(self):
        """Loads the shared initial parameters for the experiment."""
        if not os.path.exists(self.weights_filepath):
            return None
        try:
            with open(self.weights_filepath, "rb") as f:
                data = CustomUnpickler(f).load()
            return data.get("initial_params")
        except (pickle.UnpicklingError, EOFError) as e:
            logging.warning(f"Could not load initial params from {self.weights_filepath}. Error: {e}")
            return None

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

    def cleanup_live_checkpoint(self, run_key: RunKey):
        """Removes the temporary checkpoint file for a completed run."""
        filepath = self._get_resume_filepath(run_key)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                logging.error(f"Error cleaning up checkpoint for {run_key}: {e}")


def load_experiment_weights(
    experiment,
    batch_size: int,
    eta: float,
    directory: str = "experiments",
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
