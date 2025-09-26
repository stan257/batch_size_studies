import logging

import numpy as np
import pytest

from batch_size_studies.definitions import Parameterization, RunKey
from batch_size_studies.experiments import (
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
)
from batch_size_studies.synthetic_training import run_experiment


@pytest.fixture
def fixed_time_config():
    """Fixture for a fast-to-run FixedTime experiment."""
    return SyntheticExperimentFixedTime(
        D=8,
        P=32,
        N=16,
        K=2,
        num_steps=10,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
    )


@pytest.fixture
def fixed_data_config():
    """Fixture for a fast-to-run FixedData experiment."""
    return SyntheticExperimentFixedData(D=8, P=32, N=16, K=2, gamma=1.0, L=2, parameterization=Parameterization.SP)


class TestRunExperiment:
    """A test class to group all tests for the main run_experiment function."""

    def test_runs_and_returns_correct_structure(self, fixed_time_config, tmp_path):
        """
        Tests that the main training function runs without error and returns
        the expected data structures (a dict and a set).
        """
        batch_sizes = [4, 8]
        etas = [0.1]

        losses, failures = run_experiment(
            experiment=fixed_time_config,
            batch_sizes=batch_sizes,
            etas=etas,
            directory=str(tmp_path),
        )

        assert isinstance(losses, dict)
        assert isinstance(failures, set)
        assert len(losses) == 2
        assert len(failures) == 0

        # Best Practice: Use the RunKey to check for the result.
        expected_key = RunKey(batch_size=4, eta=0.1)
        assert "loss_history" in losses[expected_key]
        assert len(losses[expected_key]["loss_history"]) == fixed_time_config.num_steps

    def test_handles_failed_runs(self, fixed_time_config, tmp_path):
        """
        Tests that the training function correctly identifies and logs runs
        that fail with NaN/inf losses.
        """
        batch_sizes = [4]
        etas = [1e6]  # Use a very large learning rate to force a failure

        losses, failures = run_experiment(
            experiment=fixed_time_config,
            batch_sizes=batch_sizes,
            etas=etas,
            directory=str(tmp_path),
        )

        assert len(losses) == 0
        expected_failure = RunKey(batch_size=4, eta=1e6)
        assert failures == {expected_failure}

    def test_run_is_reproducible(self, fixed_time_config, tmp_path):
        """
        Tests that two identical training runs produce the exact same results.
        """
        experiment = fixed_time_config
        batch_sizes = [4, 8]
        etas = [0.1, 0.01]

        dir1 = tmp_path / "run1"
        dir1.mkdir()
        losses1, failed1 = run_experiment(
            experiment=experiment,
            batch_sizes=batch_sizes,
            etas=etas,
            init_key=42,
            directory=str(dir1),
        )

        dir2 = tmp_path / "run2"
        dir2.mkdir()
        losses2, failed2 = run_experiment(
            experiment=experiment,
            batch_sizes=batch_sizes,
            etas=etas,
            init_key=42,
            directory=str(dir2),
        )

        assert failed1 == failed2
        assert losses1.keys() == losses2.keys()

        for key in losses1:
            np.testing.assert_allclose(
                losses1[key]["loss_history"],
                losses2[key]["loss_history"],
                err_msg=f"Loss curves for run {key} are not identical.",
            )

    def test_run_with_fixed_data(self, fixed_data_config, tmp_path):
        num_epochs = 3
        batch_size = 8
        expected_steps = num_epochs * (fixed_data_config.P // batch_size)

        losses, _ = run_experiment(
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

    def test_raises_error_for_unknown_experiment_type(self):
        """
        Tests that a TypeError is raised if an unsupported experiment
        type is passed.
        """

        class UnknownExperiment:
            pass

        with pytest.raises(TypeError, match="Unsupported experiment type: UnknownExperiment"):
            run_experiment(experiment=UnknownExperiment(), batch_sizes=[8], etas=[0.1])

    def test_skips_run_if_batch_size_exceeds_p_for_fixed_data(self, fixed_data_config, caplog, tmp_path):
        batch_sizes = [16, 64]
        etas = [0.1]
        with caplog.at_level(logging.WARNING):
            losses, failures = run_experiment(
                experiment=fixed_data_config,
                batch_sizes=batch_sizes,
                etas=etas,
                num_epochs=1,
                directory=str(tmp_path),
            )
        valid_key = RunKey(batch_size=16, eta=0.1)
        invalid_key = RunKey(batch_size=64, eta=0.1)
        assert valid_key in losses
        assert invalid_key not in losses
        assert invalid_key not in failures
        assert "Skipping run" in caplog.text
        assert "batch_size (64) > dataset size P (32)" in caplog.text
