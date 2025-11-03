from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

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
    TrialContext,
    run_experiment_sweep,
    run_single_trial,
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

    @pytest.mark.parametrize("model_seed", [0, 42, 1337])
    def test_output_is_effectively_zero_at_initialization(self, model_seed):
        """
        Tests that the centered model's output is negligible at initialization.

        Due to JIT compilation artifacts with float32, the output of f(x) - f(x)
        may not be exactly zero. This test verifies two key properties across
        different random initializations:
        1. The centered output is negligible in an absolute sense (close to zero).
        2. The centered output's magnitude is also negligible relative to the
           uncentered output's magnitude.
        """
        data_key = jnp.ones((1, 10))

        mlp = MLP(parameterization=Parameterization.SP, gamma=1.0)
        widths = [10, 20, 5]
        params0 = mlp.init_params(model_seed, widths)
        centered_model = CenteredModel(model=mlp, params0=params0)

        uncentered_output = mlp(params0, data_key)
        centered_output = centered_model(params0, data_key)

        uncentered_norm = jnp.linalg.norm(uncentered_output)
        centered_norm = jnp.linalg.norm(centered_output)

        assert uncentered_norm > 1e-4, "Uncentered output should not be close to zero."
        # 1. Absolute check: The numerical noise should be very small.
        assert centered_norm < 1e-6, f"Centered output norm {centered_norm} is not negligible in absolute terms."
        # 2. Relative check: The centered output should be orders of magnitude smaller.
        assert centered_norm < 1e-5 * uncentered_norm, (
            "Centered output is not negligible compared to uncentered output."
        )


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

    @pytest.mark.parametrize(
        "test_id, experiment_config, result, no_save, pre_populate, expected_is_successful, expect_in_results, expect_in_failed, expect_cleanup_called",
        [
            (
                "successful_synthetic",
                {"spec": SyntheticExperimentFixedData, "is_run_complete": True},
                {"loss_history": [1.0, 0.9, 0.8]},
                False,
                False,
                True,
                True,
                False,
                True,
            ),
            (
                "incomplete_run_no_cleanup",
                {"spec": MNISTExperiment, "is_run_complete": False},
                {"loss_history": [1.0, 0.9]},
                False,
                False,
                False,
                True,
                False,
                False,
            ),
            (
                "failed_nan_accuracy",
                {"spec": MNISTExperiment, "is_run_complete": True},
                {"final_test_accuracy": np.nan},
                False,
                False,
                False,
                False,
                True,
                False,
            ),
            (
                "failed_none_result_removes_old",
                {"spec": SyntheticExperimentFixedData, "is_run_complete": False},
                None,
                True,
                True,
                False,
                False,
                True,
                False,
            ),
            (
                "successful_mnist_cleanup",
                {"spec": MNISTExperiment, "is_run_complete": True},
                {"final_test_accuracy": 0.9, "epoch_test_accuracies": [0.8, 0.85, 0.88, 0.9]},
                False,
                False,
                True,
                True,
                False,
                True,
            ),
            (
                "successful_run_no_save",
                {"spec": SyntheticExperimentFixedData, "is_run_complete": True},
                {"loss_history": [1.0, 0.9, 0.8]},
                True,
                False,
                True,
                True,
                False,
                False,
            ),
        ],
        ids=[
            "successful_synthetic_run_cleans_checkpoint",
            "incomplete_run_preserves_checkpoint",
            "failed_run_with_nan_is_not_stored",
            "failed_run_(None)_removes_previous_result",
            "successful_mnist_run_cleans_checkpoint",
            "successful_run_with_no_save_does_not_cleanup",
        ],
    )
    def test_result_validation_scenarios(
        self,
        validation_setup,
        test_id,
        experiment_config,
        result,
        no_save,
        pre_populate,
        expected_is_successful,
        expect_in_results,
        expect_in_failed,
        expect_cleanup_called,
    ):
        """Tests various scenarios for result validation, storage, and checkpoint cleanup."""
        s = validation_setup
        s.mock_exp = Mock(spec=experiment_config["spec"])
        s.mock_exp.is_run_complete.return_value = experiment_config["is_run_complete"]

        if pre_populate:
            s.results_dict[s.run_key] = {"old_data": "should be removed"}

        is_successful = validate_and_store_result(
            result=result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=no_save,
        )

        assert is_successful is expected_is_successful

        if expect_in_results:
            assert s.run_key in s.results_dict
            assert s.results_dict[s.run_key] == result
        else:
            assert s.run_key not in s.results_dict

        if expect_in_failed:
            assert s.run_key in s.failed_runs
        else:
            assert s.run_key not in s.failed_runs

        if expect_cleanup_called:
            s.checkpoint_manager.cleanup_live_checkpoint.assert_called_once_with(s.run_key)
        else:
            s.checkpoint_manager.cleanup_live_checkpoint.assert_not_called()


# ============================================================================
# TESTS FOR _run_single_trial
# ============================================================================


@patch("batch_size_studies.runner.validate_and_store_result")
@patch("batch_size_studies.runner.get_trial_runner")
@patch("batch_size_studies.runner.RunStatus")
class TestSingleTrialExecution:
    """Tests the orchestration logic for a single trial."""

    def test_skips_completed_run(self, mock_RunStatus, mock_get_runner, mock_validate):
        mock_status_instance = mock_RunStatus.return_value
        mock_status_instance.should_run = False
        mock_status_instance.is_successful = True

        # The mock context needs the attributes that RunStatus will access
        context = MagicMock(spec=TrialContext, run_key=RunKey(32, 0.1), num_steps=100, no_save=False)
        is_successful = run_single_trial(context, {}, set())

        assert is_successful is True
        mock_get_runner.assert_not_called()

    def test_runs_new_trial_successfully(self, mock_RunStatus, mock_get_runner, mock_validate):
        mock_status_instance = mock_RunStatus.return_value
        mock_status_instance.should_run = True

        mock_runner = mock_get_runner.return_value
        mock_runner.run.return_value = {"some": "result"}

        mock_validate.return_value = True  # Simulate successful validation

        # Set all attributes that will be accessed on the context object
        context = MagicMock(
            spec=TrialContext,
            run_key=RunKey(32, 0.1),
            num_steps=100,
            no_save=False,
            experiment=Mock(),
            checkpoint_manager=Mock(),
        )
        results_dict, failed_runs = {}, set()
        is_successful = run_single_trial(context, results_dict, failed_runs)

        assert is_successful is True
        mock_get_runner.assert_called_once_with(context)
        mock_runner.run.assert_called_once()
        mock_validate.assert_called_once_with(
            {"some": "result"},
            context.run_key,
            results_dict,
            failed_runs,
            context.experiment,
            context.checkpoint_manager,
            context.no_save,
        )

    def test_handles_trial_divergence(self, mock_RunStatus, mock_get_runner, mock_validate):
        mock_status_instance = mock_RunStatus.return_value
        mock_status_instance.should_run = True

        mock_runner = mock_get_runner.return_value
        mock_runner.run.return_value = None  # Simulate divergence

        mock_validate.return_value = False

        # Set all attributes that will be accessed on the context object
        context = MagicMock(
            spec=TrialContext,
            run_key=RunKey(32, 0.1),
            num_steps=100,
            no_save=False,
            experiment=Mock(),
            checkpoint_manager=Mock(),
        )
        results_dict, failed_runs = {}, set()
        is_successful = run_single_trial(context, results_dict, failed_runs)

        assert is_successful is False
        mock_runner.run.assert_called_once()
        mock_validate.assert_called_once_with(
            None,
            context.run_key,
            results_dict,
            failed_runs,
            context.experiment,
            context.checkpoint_manager,
            context.no_save,
        )


# ============================================================================
# TESTS FOR run_experiment_sweep
# ============================================================================


@patch("batch_size_studies.runner.run_single_trial")
@patch("batch_size_studies.runner._setup_sweep_state")
class TestRunExperimentSweep:
    """
    Behavioral tests for the main sweep orchestration logic.

    These tests treat `run_experiment_sweep` as a black box and assert on its
    behavior by controlling the outcome of `run_single_trial`. This is more
    robust than mocking every internal helper.
    """

    def test_sweep_runs_all_combinations(self, mock_setup, mock_run_single):
        """Verify that the sweep attempts to run every combination of hyperparameters."""
        # --- Setup ---
        # Mock the setup function to return predictable values
        mock_setup.return_value = ({}, set(), Mock(), "params0", Mock())
        mock_run_single.return_value = True

        # Define a mock experiment that requires a dataset
        mock_experiment = Mock()
        mock_experiment.prepare_datasets.return_value = ({"image": [1, 2, 3, 4]}, "test_ds")
        mock_experiment.should_skip_batch_size.return_value = False
        mock_experiment.is_online_experiment.return_value = False
        # The experiment must return a (steps, epochs) tuple
        mock_experiment.compute_num_steps.return_value = (100, 1)

        batch_sizes = [64, 128]
        etas = [0.1, 0.01]

        # --- Action ---
        run_experiment_sweep(mock_experiment, batch_sizes, etas)

        # --- Assert ---
        # It should have been called for each B x eta combination
        assert mock_run_single.call_count == 4
        calls = mock_run_single.call_args_list
        called_run_keys = {call.kwargs["context"].run_key for call in calls}
        expected_run_keys = {RunKey(64, 0.1), RunKey(64, 0.01), RunKey(128, 0.1), RunKey(128, 0.01)}
        assert called_run_keys == expected_run_keys

    def test_sweep_aborts_if_data_loading_fails(self, mock_setup, mock_run_single):
        """Verify that the sweep aborts if data loading fails for an offline experiment."""
        # --- Setup ---
        mock_setup.return_value = ({}, set(), Mock(), "params0", Mock())

        mock_experiment = Mock()
        mock_experiment.prepare_datasets.return_value = (None, None)  # Simulate data loading failure
        mock_experiment.is_online_experiment.return_value = False  # It's an offline experiment

        # --- Action ---
        results, failures = run_experiment_sweep(mock_experiment, [32], [0.1])

        # --- Assert ---
        # The main trial loop should never be called
        mock_run_single.assert_not_called()
        # The function should return the initial empty results
        assert results == {}
        assert failures == set()
