import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import CheckpointManager, load_experiment_weights
from batch_size_studies.definitions import Parameterization, RunKey
from batch_size_studies.experiments import SyntheticExperimentFixedTime


def assert_pytree_allclose(a, b):
    """Assert that two JAX PyTrees are numerically close."""
    a_flat, a_tree = jax.tree_util.tree_flatten(a)
    b_flat, b_tree = jax.tree_util.tree_flatten(b)
    assert a_tree == b_tree, "PyTree structures do not match"
    for arr_a, arr_b in zip(a_flat, b_flat):
        np.testing.assert_allclose(arr_a, arr_b, rtol=1e-5)


@pytest.fixture
def experiment_instance():
    """A fixture for a sample experiment dataclass."""
    return SyntheticExperimentFixedTime(
        D=10,
        P=100,
        N=32,
        K=2,
        num_steps=1000,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
    )


@pytest.fixture
def checkpoint_manager(experiment_instance, tmp_path):
    """A fixture that provides a CheckpointManager in a temporary directory."""
    return CheckpointManager(experiment_instance, directory=str(tmp_path))


@pytest.fixture
def mock_data():
    """A fixture for mock JAX parameters and optimizer state."""
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    params = [jax.random.normal(key1, (10, 5)), jax.random.normal(key2, (5, 2))]
    opt_state = (jnp.zeros_like(params[0]), jnp.zeros_like(params[1]))
    return params, opt_state


class TestCheckpointManager:
    def test_initialization(self, checkpoint_manager, experiment_instance, tmp_path):
        """Tests that the manager creates the correct paths and directories."""
        exp_dir = tmp_path / experiment_instance.experiment_type
        assert os.path.isdir(checkpoint_manager.checkpoint_dir)
        assert checkpoint_manager.checkpoint_dir.startswith(str(exp_dir))
        assert checkpoint_manager.weights_filepath.startswith(str(exp_dir))

    def test_save_and_load_live_checkpoint(self, checkpoint_manager, mock_data):
        """Tests the full save-and-load cycle for a live checkpoint."""
        run_key = RunKey(batch_size=16, eta=0.1)
        params, opt_state = mock_data
        results = {"loss_history": [0.5, 0.4, 0.3]}
        step = 2

        # Save
        checkpoint_manager.save_live_checkpoint(run_key, step, params, opt_state, results)

        # Load
        loaded_params, loaded_opt_state, loaded_results, loaded_step = checkpoint_manager.load_live_checkpoint(run_key)

        # Assert correctness
        assert loaded_step == step + 1
        assert loaded_results == results
        assert_pytree_allclose(loaded_params, params)
        assert_pytree_allclose(loaded_opt_state, opt_state)

    def test_load_nonexistent_live_checkpoint(self, checkpoint_manager):
        """Tests that loading a non-existent checkpoint returns correct defaults."""
        run_key = RunKey(batch_size=16, eta=0.1)
        params, opt_state, results, start_step = checkpoint_manager.load_live_checkpoint(run_key)

        assert params is None
        assert opt_state is None
        assert results == {}
        assert start_step == 0

    def test_load_corrupted_live_checkpoint(self, checkpoint_manager, caplog):
        """Edge Case: Tests graceful failure when a checkpoint file is corrupted."""
        run_key = RunKey(batch_size=32, eta=0.01)
        filepath = checkpoint_manager._get_resume_filepath(run_key)

        # Write garbage to the file
        with open(filepath, "w") as f:
            f.write("this is not a valid pickle file")

        params, opt_state, loss_history, start_step = checkpoint_manager.load_live_checkpoint(run_key)

        # Assert that it fails gracefully and returns defaults
        assert params is None
        assert start_step == 0
        assert "Warning: Could not load checkpoint" in caplog.text

    def test_analysis_snapshot_creation_and_delta_reconstruction(self, checkpoint_manager, mock_data):
        """Tests saving the first snapshot and reconstructing weights from the delta."""
        run_key = RunKey(batch_size=16, eta=0.1)
        params, _ = mock_data
        # Create a slightly different initial_params
        initial_params = jax.tree_util.tree_map(lambda x: x - 0.1, params)

        # Save the first snapshot
        checkpoint_manager.save_analysis_snapshot(run_key, step=100, params=params, initial_params=initial_params)

        assert os.path.exists(checkpoint_manager.weights_filepath)

        # Load the data back
        with open(checkpoint_manager.weights_filepath, "rb") as f:
            weights_data = pickle.load(f)

        # Verify initial params and the delta snapshot
        assert_pytree_allclose(weights_data["initial_params"], initial_params)
        delta = weights_data["weight_snapshots"][run_key][100]

        # The crucial test: reconstruct the original params and verify
        reconstructed_params = jax.tree_util.tree_map(lambda p0, d: p0 + d, initial_params, delta)
        assert_pytree_allclose(reconstructed_params, params)

    def test_analysis_snapshot_append(self, checkpoint_manager, mock_data):
        """Tests that subsequent saves append to the snapshot file correctly."""
        params, _ = mock_data
        initial_params = jax.tree_util.tree_map(lambda x: x - 0.1, params)

        # Save for first run
        checkpoint_manager.save_analysis_snapshot(RunKey(16, 0.1), 100, params, initial_params)
        # Save for a second, different run
        checkpoint_manager.save_analysis_snapshot(RunKey(32, 0.01), 200, params, initial_params)

        with open(checkpoint_manager.weights_filepath, "rb") as f:
            weights_data = pickle.load(f)

        # Check that both snapshots are present
        snapshots = weights_data["weight_snapshots"]
        assert RunKey(16, 0.1) in snapshots
        assert 100 in snapshots[RunKey(16, 0.1)]
        assert RunKey(32, 0.01) in snapshots
        assert 200 in snapshots[RunKey(32, 0.01)]
        # Check that initial_params were not overwritten
        assert_pytree_allclose(weights_data["initial_params"], initial_params)

    def test_cleanup_live_checkpoint(self, checkpoint_manager, mock_data):
        """Tests that the cleanup method removes the resume checkpoint file."""
        run_key = RunKey(batch_size=16, eta=0.1)
        params, opt_state = mock_data
        filepath = checkpoint_manager._get_resume_filepath(run_key)

        # Save a checkpoint and confirm it exists
        checkpoint_manager.save_live_checkpoint(run_key, step=1, params=params, opt_state=opt_state, results={})
        assert os.path.exists(filepath)

        # Clean it up
        checkpoint_manager.cleanup_live_checkpoint(run_key)

        # Assert it's gone
        assert not os.path.exists(filepath)

    def test_load_initial_params(self, checkpoint_manager, mock_data):
        """Tests loading of initial_params from the analysis snapshot file."""
        # 1. Before file exists, should return None
        assert checkpoint_manager.load_initial_params() is None

        # 2. After file is created, should return the params
        params, _ = mock_data
        initial_params = jax.tree_util.tree_map(lambda x: x - 0.1, params)
        checkpoint_manager.save_analysis_snapshot(RunKey(16, 0.1), 100, params, initial_params)

        loaded_initial_params = checkpoint_manager.load_initial_params()
        assert_pytree_allclose(loaded_initial_params, initial_params)

    def test_resume_filepath_generation(self, checkpoint_manager):
        """Tests that the resume checkpoint filename is generated correctly."""
        run_key = RunKey(batch_size=128, eta=0.05)
        filepath = checkpoint_manager._get_resume_filepath(run_key)

        # Check that the filename correctly replaces '.' with 'p' for the float
        assert "B=128_eta=0p05.pkl" in filepath

        # Check that it's in the right directory
        assert filepath.startswith(checkpoint_manager.checkpoint_dir)

    def test_load_full_weight_history(self, checkpoint_manager, mock_data):
        """Tests loading the entire history of weight snapshots for a run."""
        run_key = RunKey(batch_size=16, eta=0.1)
        params, _ = mock_data
        initial_params = jax.tree_util.tree_map(lambda x: x - 0.1, params)

        # Create a second set of params for another step
        params_step200 = jax.tree_util.tree_map(lambda x: x + 0.5, params)

        # Save two snapshots for the same run
        checkpoint_manager.save_analysis_snapshot(run_key, step=100, params=params, initial_params=initial_params)
        checkpoint_manager.save_analysis_snapshot(
            run_key, step=200, params=params_step200, initial_params=initial_params
        )

        # Load the full history
        history = checkpoint_manager.load_full_weight_history(run_key)

        assert history is not None
        assert isinstance(history, dict)
        assert set(history.keys()) == {100, 200}

        # Verify the reconstruction for both steps
        assert_pytree_allclose(history[100], params)
        assert_pytree_allclose(history[200], params_step200)

    def test_load_full_weight_history_not_found(self, checkpoint_manager):
        """Tests that loading history for a non-existent run returns an empty dict."""
        run_key = RunKey(batch_size=99, eta=0.99)
        history = checkpoint_manager.load_full_weight_history(run_key)
        assert history == {}

    def test_initial_weights_consistency(self, checkpoint_manager, mock_data):
        """
        Tests that loading initial weights via different methods yields the same result.
        - load_initial_params()
        - load_analysis_snapshot() at step 0
        - load_full_weight_history()[0]
        """
        run_key = RunKey(batch_size=16, eta=0.1)
        initial_params, _ = mock_data

        # At step 0, the current params are the same as the initial params.
        checkpoint_manager.save_analysis_snapshot(run_key, step=0, params=initial_params, initial_params=initial_params)

        # Method 1: Direct loading
        loaded_initial = checkpoint_manager.load_initial_params()

        # Method 2: Loading as a single snapshot at step 0
        snapshot_at_0 = checkpoint_manager.load_analysis_snapshot(run_key, 0)

        # Method 3: Loading from the full history
        history = checkpoint_manager.load_full_weight_history(run_key)
        history_at_0 = history[0]

        assert_pytree_allclose(loaded_initial, snapshot_at_0)
        assert_pytree_allclose(loaded_initial, history_at_0)

    def test_initial_weights_are_shared_across_runs(self, checkpoint_manager, mock_data):
        """
        Tests that different runs (i.e only change in (B,η)) within the same experiment share the same
        initial weights file and do not overwrite it.
        """
        initial_params, _ = mock_data

        # Save a snapshot for a first run
        checkpoint_manager.save_analysis_snapshot(RunKey(16, 0.1), 100, initial_params, initial_params)
        # Save a snapshot for a second run
        checkpoint_manager.save_analysis_snapshot(RunKey(32, 0.01), 100, initial_params, initial_params)

        # Load the initial params after both runs have saved.
        # It should still be the original initial_params.
        loaded_initial = checkpoint_manager.load_initial_params()
        assert_pytree_allclose(loaded_initial, initial_params)


class TestLoadExperimentWeights:
    """Tests the high-level load_experiment_weights utility function."""

    @pytest.fixture
    def setup_weights_file(self, checkpoint_manager, mock_data):
        """A fixture to create a sample weights file with a few snapshots."""
        run_key = RunKey(batch_size=16, eta=0.1)
        params, _ = mock_data
        initial_params = jax.tree_util.tree_map(lambda x: x - 0.1, params)
        params_step200 = jax.tree_util.tree_map(lambda x: x + 0.5, params)

        checkpoint_manager.save_analysis_snapshot(run_key, step=100, params=params, initial_params=initial_params)
        checkpoint_manager.save_analysis_snapshot(
            run_key, step=200, params=params_step200, initial_params=initial_params
        )
        return params, params_step200

    def test_load_single_step(self, experiment_instance, setup_weights_file, tmp_path):
        """Tests loading weights for a single, specific step."""
        original_params, _ = setup_weights_file

        loaded_params = load_experiment_weights(
            experiment=experiment_instance,
            batch_size=16,
            eta=0.1,
            directory=str(tmp_path),
            step_to_load=100,
        )

        assert loaded_params is not None
        assert_pytree_allclose(loaded_params, original_params)

    def test_load_full_history(self, experiment_instance, setup_weights_file, tmp_path):
        """Tests loading the full history of weights for a run."""
        _, params_step200 = setup_weights_file

        history = load_experiment_weights(
            experiment=experiment_instance,
            batch_size=16,
            eta=0.1,
            directory=str(tmp_path),
        )

        assert isinstance(history, dict)
        assert set(history.keys()) == {100, 200}
        assert_pytree_allclose(history[200], params_step200)
