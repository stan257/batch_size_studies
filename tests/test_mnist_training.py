import logging
import os

import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import RunKey
from batch_size_studies.experiments import MNISTExperiment, OptimizerType, Parameterization
from batch_size_studies.mnist_training import run_mnist_experiment

# --- Fixtures ---


@pytest.fixture
def mnist_config():
    """Fixture for a fast-to-run MNIST experiment."""
    return MNISTExperiment(
        N=32,
        L=2,
        num_epochs=4,  # Use an even number for easy splitting
        parameterization=Parameterization.SP,
    )


@pytest.fixture
def mock_mnist_loader():
    """
    A mock dataset loader that returns small numpy arrays,
    avoiding the need to download the actual MNIST dataset.
    """

    def _loader():
        # Using a fixed seed for numpy's random generator makes the test deterministic.
        np.random.seed(42)
        train_images = np.random.rand(128, 28, 28, 1).astype(np.float32)
        train_labels = np.random.randint(0, 10, 128).astype(np.int32)
        test_images = np.random.rand(64, 28, 28, 1).astype(np.float32)
        test_labels = np.random.randint(0, 10, 64).astype(np.int32)
        return (train_images, train_labels), (test_images, test_labels)

    return _loader


# --- Test Class ---


class TestMNISTExperiment:
    def test_runs_and_returns_correct_structure(self, mnist_config, mock_mnist_loader, tmp_path):
        """Tests that the MNIST runner completes and returns the correct structure."""
        batch_sizes = [32]
        etas = [0.01]

        results, failures = run_mnist_experiment(
            experiment=mnist_config,
            batch_sizes=batch_sizes,
            etas=etas,
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path),
        )

        assert isinstance(results, dict)
        assert isinstance(failures, set)
        assert len(failures) == 0
        assert len(results) == 1

        run_key = RunKey(batch_size=32, eta=0.01)
        assert run_key in results
        assert "final_test_accuracy" in results[run_key]
        assert len(results[run_key]["epoch_test_accuracies"]) == mnist_config.num_epochs

    def test_checkpoint_and_resume(self, mnist_config, mock_mnist_loader, tmp_path, caplog):
        """
        Tests that an interrupted MNIST experiment correctly resumes from the last
        completed epoch.
        """
        batch_sizes = [64]
        etas = [0.01]
        total_epochs = mnist_config.num_epochs  # 4
        resume_from_epoch = 2

        # --- 1. Run the experiment partway ---
        run_mnist_experiment(
            experiment=mnist_config,
            batch_sizes=batch_sizes,
            etas=etas,
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path),
            num_epochs=resume_from_epoch,  # Run for 2 epochs
        )

        # Check that a checkpoint file was created
        cm = CheckpointManager(mnist_config, directory=str(tmp_path))
        run_key = RunKey(batch_size=64, eta=0.01)
        resume_file = cm._get_resume_filepath(run_key)
        assert os.path.exists(resume_file)

        # --- 2. Run the experiment again to completion ---
        caplog.clear()
        with caplog.at_level(logging.INFO):
            results, failures = run_mnist_experiment(
                experiment=mnist_config,
                batch_sizes=batch_sizes,
                etas=etas,
                dataset_loader=mock_mnist_loader,
                directory=str(tmp_path),
                num_epochs=total_epochs,  # Now run for the full 4 epochs
            )

        # --- 3. Verify the results ---
        # Check that the log shows it resumed
        assert f"Resuming run {run_key} from step {resume_from_epoch}" in caplog.text

        # Check that the final result has the correct number of epochs
        assert len(results[run_key]["epoch_test_accuracies"]) == total_epochs

        # Check that the temporary checkpoint file was cleaned up on completion
        assert not os.path.exists(resume_file)

    def test_handles_failed_runs(self, mnist_config, mock_mnist_loader, tmp_path):
        """Tests that a run that diverges is correctly marked as failed."""
        batch_sizes = [32]
        etas = [1e20]  # A huge learning rate to force divergence

        results, failures = run_mnist_experiment(
            experiment=mnist_config,
            batch_sizes=batch_sizes,
            etas=etas,
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path),
        )

        assert len(results) == 0
        assert len(failures) == 1
        assert RunKey(batch_size=32, eta=1e20) in failures

    def test_optimizer_selection_works(self, mnist_config, mock_mnist_loader, tmp_path):
        """
        Tests that changing the optimizer in the config results in a different
        training outcome.
        """
        from dataclasses import replace

        import jax

        np.random.seed(257)
        batch_sizes = [64]
        etas = [0.1]

        # Run with SGD
        sgd_config = mnist_config
        results_sgd, _ = run_mnist_experiment(
            experiment=sgd_config,
            batch_sizes=batch_sizes,
            etas=etas,
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path / "sgd"),
        )

        # Run with Adam
        adam_config = replace(mnist_config, optimizer=OptimizerType.ADAM)
        results_adam, _ = run_mnist_experiment(
            experiment=adam_config,
            batch_sizes=batch_sizes,
            etas=etas,
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path / "adam"),
        )

        # A robust test to check if the final model *parameters* are different
        run_key = RunKey(batch_size=64, eta=0.1)

        # Load the final model parameters from the analysis snapshots, which are
        # designed for post-hoc analysis and are not cleaned up.
        cm_sgd = CheckpointManager(sgd_config, directory=str(tmp_path / "sgd"))
        last_epoch = sgd_config.num_epochs - 1
        params_sgd = cm_sgd.load_analysis_snapshot(run_key, step=last_epoch)

        cm_adam = CheckpointManager(adam_config, directory=str(tmp_path / "adam"))
        params_adam = cm_adam.load_analysis_snapshot(run_key, step=last_epoch)

        assert params_sgd is not None, "SGD run did not produce a checkpoint."
        assert params_adam is not None, "Adam run did not produce a checkpoint."

        # Assert that the final parameters are not numerically close.
        sgd_leaves, _ = jax.tree_util.tree_flatten(params_sgd)
        adam_leaves, _ = jax.tree_util.tree_flatten(params_adam)

        are_different = any(
            not np.allclose(sgd_leaf, adam_leaf) for sgd_leaf, adam_leaf in zip(sgd_leaves, adam_leaves)
        )

        assert are_different, "Final model parameters for SGD and Adam were unexpectedly identical."
