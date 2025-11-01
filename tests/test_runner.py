from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import MNISTExperiment, SyntheticExperimentFixedData
from batch_size_studies.models import MLP
from batch_size_studies.runner import (
    CenteredModel,
    EtaStabilityTracker,
    RunStatus,
    validate_and_store_result,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_run_key():
    """Provides a mock RunKey with default values."""
    mock_key = Mock(spec=RunKey)
    mock_key.batch_size = 32
    mock_key.eta = 0.01
    return mock_key


@pytest.fixture
def mock_experiment():
    """Provides a generic mock experiment with common attributes."""
    # Use a real experiment type to satisfy isinstance checks
    mock_exp = Mock(spec=MNISTExperiment)
    mock_exp.gamma = 1.0
    mock_exp.L = 2
    mock_exp.optimizer = OptimizerType.SGD
    mock_exp.parameterization = Parameterization.SP
    mock_exp.N = 128
    mock_exp.loss_type = LossType.XENT
    mock_exp.num_outputs = 10
    mock_exp.P = 1000
    return mock_exp


@pytest.fixture
def validation_setup():
    """Provides a standard setup for testing validate_and_store_result."""
    mock_exp = Mock(spec=SyntheticExperimentFixedData, num_epochs=4)

    checkpoint_manager = Mock()
    checkpoint_manager.exp_dir = "/fake/path"

    return SimpleNamespace(
        mock_exp=mock_exp,
        run_key=RunKey(batch_size=32, eta=0.1),
        result={"loss_history": [1.0, 0.9, 0.8]},
        results_dict={},
        failed_runs=set(),
        checkpoint_manager=checkpoint_manager,
    )


# ============================================================================
# TESTS FOR CenteredModel
# ============================================================================


class TestCenteredModel:
    """Tests for the CenteredModel wrapper class."""

    def test_output_is_zero_at_initialization(self):
        model_seed = 0
        data_key = jnp.ones((1, 10))

        mlp = MLP(parameterization=Parameterization.SP, gamma=1.0)
        widths = [10, 20, 5]
        params0 = mlp.init_params(model_seed, widths)

        centered_model = CenteredModel(model=mlp, params0=params0)

        uncentered_output = mlp(params0, data_key)
        assert uncentered_output.shape == (1, 5)
        assert not jnp.all(uncentered_output == 0), "Uncentered model output should not be zero at initialization"

        centered_output = centered_model(params0, data_key)
        # The output should be a zero vector of the correct shape
        assert centered_output.shape == (1, 5)
        assert jnp.all(centered_output == 0)


# ============================================================================
# TESTS FOR EtaStabilityTracker
# ============================================================================


class TestEtaStabilityTracker:
    """Tests the logic for early stopping based on consecutive successes."""

    def test_threshold_reached_after_consecutive_successes(self):
        tracker = EtaStabilityTracker(depth=3)

        assert tracker.update(is_successful=True) is False
        assert tracker.count == 1

        assert tracker.update(is_successful=True) is False
        assert tracker.count == 2

        assert tracker.update(is_successful=True) is True
        assert tracker.count == 3

    def test_failure_resets_counter(self):
        tracker = EtaStabilityTracker(depth=3)

        tracker.update(is_successful=True)
        tracker.update(is_successful=True)
        assert tracker.count == 2

        tracker.update(is_successful=False)
        assert tracker.count == 0

        tracker.update(is_successful=True)
        tracker.update(is_successful=True)
        assert tracker.update(is_successful=True) is True

    def test_disabled_when_depth_is_none(self):
        tracker = EtaStabilityTracker(depth=None)

        # Should never trigger regardless of successes
        for _ in range(10):
            assert tracker.update(is_successful=True) is False

    def test_disabled_when_depth_is_zero(self):
        tracker = EtaStabilityTracker(depth=0)

        assert tracker.update(is_successful=True) is False

    def test_reset_clears_counter(self):
        tracker = EtaStabilityTracker(depth=3)

        tracker.update(is_successful=True)
        tracker.update(is_successful=True)
        assert tracker.count == 2

        tracker.reset()
        assert tracker.count == 0

    def test_handles_alternating_success_failure(self):
        tracker = EtaStabilityTracker(depth=3)

        # Alternating pattern should never reach threshold
        for _ in range(10):
            assert tracker.update(is_successful=True) is False
            tracker.update(is_successful=False)

        assert tracker.count == 0


# ============================================================================
# TESTS FOR RunStatus
# ============================================================================


class TestRunStatus:
    """Tests the logic for determining if a trial should be run or skipped."""

    def test_should_run_when_no_save_enabled(self):
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=True
        )

        assert status.should_run is True
        assert status.is_successful is False

    def test_skip_previously_failed_run(self):
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {}
        failed_runs = {run_key}

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is False
        assert status.is_successful is False

    def test_skip_completed_run(self):
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {run_key: {"loss_history": [1.0] * 1000}}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is False
        assert status.is_successful is True

    def test_run_incomplete_result(self):
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {run_key: {"loss_history": [1.0] * 500}}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is True

    def test_run_missing_loss_history(self):
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {run_key: {"other_metric": 99}}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is True


# ============================================================================
# TESTS FOR validate_and_store_result
# ============================================================================


class TestValidateAndStoreResult:
    """Tests the logic for validating and storing trial results."""

    def test_successful_synthetic_result_stored(self, validation_setup):
        s = validation_setup
        s.mock_exp.is_run_complete.return_value = True  # Simulate a complete run

        is_successful = validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        assert is_successful is True
        assert s.run_key in s.results_dict
        assert s.results_dict[s.run_key] == s.result
        assert s.run_key not in s.failed_runs

    def test_incomplete_run_is_stored_for_resumption(self, validation_setup):
        """
        Tests that an incomplete but valid run (e.g., finished early) is stored
        in the results dictionary but NOT marked as a failure. This is the
        correct behavior to allow for checkpoint resumption.
        """
        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment)
        s.mock_exp.is_run_complete.return_value = False  # Mark as incomplete
        s.result = {"loss_history": [1.0, 0.9, 0.8]}  # A valid, partial result

        is_successful = validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        # The run is not "successful" because it's not complete
        assert is_successful is False
        # CRUCIAL: The result should be stored to allow for resumption...
        assert s.run_key in s.results_dict
        # ...but it should NOT be added to the set of hard failures.
        assert s.run_key not in s.failed_runs

    def test_mnist_result_with_nan_accuracy(self, validation_setup):
        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment)
        s.mock_exp.is_run_complete.return_value = True  # Structurally complete
        s.result = {"final_test_accuracy": np.nan}

        is_successful = validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        assert is_successful is False
        assert s.run_key in s.failed_runs

    def test_checkpoint_cleanup_called_for_completed_runs(self, validation_setup):
        s = validation_setup

        s.mock_exp.is_run_complete.return_value = True
        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=False,  # Enable saving
        )

        # Should cleanup checkpoint for successful synthetic run
        s.checkpoint_manager.cleanup_live_checkpoint.assert_called_once_with(s.run_key)

    def test_checkpoint_cleanup_called_for_full_mnist_run(self, validation_setup):
        """Test that checkpoints ARE cleaned up for fully completed MNIST runs."""
        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment, num_epochs=4)
        s.mock_exp.is_run_complete.return_value = True
        s.result = {"final_test_accuracy": 0.9, "epoch_test_accuracies": [0.8, 0.85, 0.88, 0.9]}

        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=False,
        )

        # Should BE called because run is fully complete
        s.checkpoint_manager.cleanup_live_checkpoint.assert_called_once_with(s.run_key)

    def test_checkpoint_cleanup_not_called_for_partial_mnist_run(self, validation_setup):
        """Test that checkpoints are NOT cleaned up for partially completed MNIST runs."""
        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment, num_epochs=4)
        s.mock_exp.is_run_complete.return_value = False  # This run is incomplete
        s.result = {"final_test_accuracy": 0.9, "epoch_test_accuracies": [0.8, 0.85]}

        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=False,
        )

        # Should NOT be called because run is not fully complete
        s.checkpoint_manager.cleanup_live_checkpoint.assert_not_called()

    def test_previous_result_removed_on_failure(self, validation_setup):
        s = validation_setup
        s.mock_exp.is_run_complete.return_value = False
        s.results_dict[s.run_key] = {"old_result": "data"}

        validate_and_store_result(
            result=None,  # Failed run
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        assert s.run_key not in s.results_dict
        assert s.run_key in s.failed_runs
