import logging
import os
from dataclasses import dataclass

import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import (
    ExperimentBase,
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
)
from batch_size_studies.runner import run_experiment_sweep

# --- Fixtures ---


@pytest.fixture
def fixed_time_config():
    """Fixture for a fast-to-run FixedTime experiment."""
    return SyntheticExperimentFixedTime(
        D=8, P=32, N=16, K=2, num_steps=10, gamma=1.0, L=2, parameterization=Parameterization.SP
    )


@pytest.fixture
def fixed_data_config():
    """Fixture for a fast-to-run FixedData experiment."""
    return SyntheticExperimentFixedData(D=8, P=32, N=16, K=2, gamma=1.0, L=2, parameterization=Parameterization.SP)


@pytest.fixture
def mnist_config():
    """Fixture for a fast-to-run MNIST experiment."""
    return MNISTExperiment(N=32, L=2, num_epochs=4, parameterization=Parameterization.SP)


@pytest.fixture
def mock_mnist_loader():
    """A mock dataset loader that returns small numpy arrays."""

    def _loader():
        np.random.seed(42)
        train_images = np.random.rand(128, 28, 28, 1).astype(np.float32)
        train_labels = np.random.randint(0, 10, 128).astype(np.int32)
        test_images = np.random.rand(64, 28, 28, 1).astype(np.float32)
        test_labels = np.random.randint(0, 10, 64).astype(np.int32)
        return (train_images, train_labels), (test_images, test_labels)

    return _loader


# --- Test Classes ---


class TestUnifiedRunner:
    def test_handles_unknown_experiment_type(self, tmp_path, caplog):
        """Tests that the runner handles an unknown experiment type gracefully."""

        @dataclass
        class UnknownExperiment(ExperimentBase):
            # Add attributes to satisfy the runner's common setup logic
            experiment_type: str = "unknown"
            parameterization: Parameterization = Parameterization.SP
            gamma: float = 1.0
            D: int = 10
            N: int = 16
            L: int = 2
            P: int = 1000

        config = UnknownExperiment()
        with caplog.at_level(logging.INFO):
            losses, failures = run_experiment_sweep(
                experiment=config, batch_sizes=[8], etas=[0.1], directory=str(tmp_path)
            )

        assert len(losses) == 0
        assert len(failures) == 1
        assert RunKey(8, 0.1) in failures


class TestSyntheticRunner:
    def test_runs_and_returns_correct_structure(self, fixed_time_config, tmp_path):
        """Tests that the main training function runs without error and returns the expected data structures."""
        losses, failures = run_experiment_sweep(
            experiment=fixed_time_config, batch_sizes=[4, 8], etas=[0.1], directory=str(tmp_path)
        )
        assert isinstance(losses, dict)
        assert isinstance(failures, set)
        assert len(losses) == 2
        assert len(failures) == 0
        expected_key = RunKey(batch_size=4, eta=0.1)
        assert "loss_history" in losses[expected_key]
        assert len(losses[expected_key]["loss_history"]) == fixed_time_config.num_steps

    def test_handles_failed_runs(self, fixed_time_config, tmp_path):
        """Tests that the training function correctly identifies and logs runs that fail with NaN/inf losses."""
        losses, failures = run_experiment_sweep(
            experiment=fixed_time_config, batch_sizes=[4], etas=[1e6], directory=str(tmp_path)
        )
        assert len(losses) == 0
        assert failures == {RunKey(batch_size=4, eta=1e6)}

    def test_run_is_reproducible(self, fixed_time_config, tmp_path):
        """Tests that two identical training runs produce the exact same results."""
        losses1, failed1 = run_experiment_sweep(
            experiment=fixed_time_config,
            batch_sizes=[4, 8],
            etas=[0.1, 0.01],
            init_key=42,
            directory=str(tmp_path / "run1"),
        )
        losses2, failed2 = run_experiment_sweep(
            experiment=fixed_time_config,
            batch_sizes=[4, 8],
            etas=[0.1, 0.01],
            init_key=42,
            directory=str(tmp_path / "run2"),
        )
        assert failed1 == failed2
        assert losses1.keys() == losses2.keys()
        for key in losses1:
            np.testing.assert_allclose(losses1[key]["loss_history"], losses2[key]["loss_history"])

    def test_run_with_fixed_data(self, fixed_data_config, tmp_path):
        num_epochs, batch_size = 3, 8
        expected_steps = num_epochs * (fixed_data_config.P // batch_size)
        losses, _ = run_experiment_sweep(
            experiment=fixed_data_config,
            batch_sizes=[batch_size],
            etas=[0.1],
            init_key=0,
            num_epochs=num_epochs,
            directory=str(tmp_path),
        )
        expected_key = RunKey(batch_size=batch_size, eta=0.1)
        assert "loss_history" in losses[expected_key]
        assert len(losses[expected_key]["loss_history"]) == expected_steps

    def test_skips_run_if_batch_size_exceeds_p_for_fixed_data(self, fixed_data_config, caplog, tmp_path):
        with caplog.at_level(logging.WARNING):
            losses, failures = run_experiment_sweep(
                experiment=fixed_data_config, batch_sizes=[16, 64], etas=[0.1], num_epochs=1, directory=str(tmp_path)
            )
        assert RunKey(16, 0.1) in losses
        assert RunKey(64, 0.1) not in losses
        assert RunKey(64, 0.1) not in failures
        assert "Skipping run" in caplog.text and "batch_size (64) > dataset size (32)" in caplog.text


class TestMNISTRunner:
    def test_runs_and_returns_correct_structure(self, mnist_config, mock_mnist_loader, tmp_path):
        """Tests that the MNIST runner completes and returns the correct structure."""
        results, failures = run_experiment_sweep(
            experiment=mnist_config,
            batch_sizes=[32],
            etas=[0.01],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path),
        )
        assert isinstance(results, dict) and isinstance(failures, set)
        assert len(failures) == 0 and len(results) == 1
        run_key = RunKey(batch_size=32, eta=0.01)
        assert run_key in results
        assert "final_test_accuracy" in results[run_key]
        assert len(results[run_key]["epoch_test_accuracies"]) == mnist_config.num_epochs

    def test_checkpoint_and_resume(self, mnist_config, mock_mnist_loader, tmp_path, caplog):
        """Tests that an interrupted MNIST experiment correctly resumes from the last completed epoch."""
        total_epochs, resume_from_epoch = mnist_config.num_epochs, 2
        run_key = RunKey(batch_size=64, eta=0.01)

        # Run partway
        run_experiment_sweep(
            experiment=mnist_config,
            batch_sizes=[64],
            etas=[0.01],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path),
            num_epochs=resume_from_epoch,
        )
        cm = CheckpointManager(mnist_config, directory=str(tmp_path))
        resume_file = cm._get_resume_filepath(run_key)
        assert os.path.exists(resume_file)

        # Run to completion
        caplog.clear()
        with caplog.at_level(logging.INFO):
            results, _ = run_experiment_sweep(
                experiment=mnist_config,
                batch_sizes=[64],
                etas=[0.01],
                dataset_loader=mock_mnist_loader,
                directory=str(tmp_path),
                num_epochs=total_epochs,
            )

        assert f"Resuming run {run_key} from step {resume_from_epoch}" in caplog.text
        assert len(results[run_key]["epoch_test_accuracies"]) == total_epochs
        assert not os.path.exists(resume_file)

    def test_handles_failed_runs(self, mnist_config, mock_mnist_loader, tmp_path):
        """Tests that a run that diverges is correctly marked as failed."""
        _, failures = run_experiment_sweep(
            experiment=mnist_config,
            batch_sizes=[32],
            etas=[1e20],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path),
        )
        assert len(failures) == 1 and RunKey(batch_size=32, eta=1e20) in failures

    def test_optimizer_selection_works(self, mnist_config, mock_mnist_loader, tmp_path):
        """Tests that changing the optimizer in the config results in a different training outcome."""
        from dataclasses import replace

        import jax

        run_key = RunKey(batch_size=64, eta=0.1)
        last_epoch = mnist_config.num_epochs - 1

        # Run with SGD
        sgd_config = mnist_config
        run_experiment_sweep(
            experiment=sgd_config,
            batch_sizes=[64],
            etas=[0.1],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path / "sgd"),
        )
        cm_sgd = CheckpointManager(sgd_config, directory=str(tmp_path / "sgd"))
        params_sgd = cm_sgd.load_analysis_snapshot(run_key, step=last_epoch)

        # Run with Adam
        adam_config = replace(mnist_config, optimizer=OptimizerType.ADAM)
        run_experiment_sweep(
            experiment=adam_config,
            batch_sizes=[64],
            etas=[0.1],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path / "adam"),
        )
        cm_adam = CheckpointManager(adam_config, directory=str(tmp_path / "adam"))
        params_adam = cm_adam.load_analysis_snapshot(run_key, step=last_epoch)

        assert params_sgd is not None and params_adam is not None
        sgd_leaves, _ = jax.tree_util.tree_flatten(params_sgd)
        adam_leaves, _ = jax.tree_util.tree_flatten(params_adam)
        are_different = any(not np.allclose(s, a) for s, a in zip(sgd_leaves, adam_leaves))
        assert are_different, "Final model parameters for SGD and Adam were unexpectedly identical."
