from unittest.mock import MagicMock, Mock, patch

import pytest

from batch_size_studies.definitions import RunKey
from batch_size_studies.experiments import ExperimentBase
from batch_size_studies.runner import (
    TrialContext,
    _is_run_complete,
    _validate_and_store_partial_result,
    run_single_trial,
)


class Test_validate_and_store_partial_result:
    """Tests for the focused validation and storage helper function."""

    @pytest.fixture
    def setup(self):
        mock_exp = Mock(spec=ExperimentBase)
        run_key = RunKey(32, 0.1)
        results_dict = {}
        failed_runs = set()
        directory = "/fake/dir"
        return locals()

    def test_valid_result_is_stored(self, setup):
        s = setup
        result = {"loss_history": [1.0]}

        is_valid = _validate_and_store_partial_result(
            result, s["run_key"], s["results_dict"], s["failed_runs"], s["mock_exp"], False, s["directory"]
        )

        assert is_valid is True
        assert s["run_key"] in s["results_dict"]
        assert s["results_dict"][s["run_key"]] == result
        assert s["run_key"] not in s["failed_runs"]
        s["mock_exp"].save_results.assert_called_once_with(s["results_dict"], s["failed_runs"], s["directory"])

    def test_invalid_result_is_marked_as_failed(self, setup):
        s = setup
        s["results_dict"][s["run_key"]] = {"loss_history": [1.0]}  # Pre-populate

        is_valid = _validate_and_store_partial_result(
            None, s["run_key"], s["results_dict"], s["failed_runs"], s["mock_exp"], False, s["directory"]
        )

        assert is_valid is False
        assert s["run_key"] not in s["results_dict"]  # Should be removed
        assert s["run_key"] in s["failed_runs"]
        s["mock_exp"].save_results.assert_called_once_with(s["results_dict"], s["failed_runs"], s["directory"])

    def test_no_save_mode_does_not_save(self, setup):
        s = setup
        result = {"loss_history": [1.0]}

        _validate_and_store_partial_result(
            result, s["run_key"], s["results_dict"], s["failed_runs"], s["mock_exp"], True, s["directory"]
        )

        s["mock_exp"].save_results.assert_not_called()

    def test_valid_result_removes_from_failed_set(self, setup):
        s = setup
        s["failed_runs"].add(s["run_key"])  # Pre-populate as failed
        result = {"loss_history": [1.0]}

        is_valid = _validate_and_store_partial_result(
            result, s["run_key"], s["results_dict"], s["failed_runs"], s["mock_exp"], False, s["directory"]
        )

        assert is_valid is True
        assert s["run_key"] not in s["failed_runs"]


class Test_is_run_complete:
    """Tests for the focused completion checking helper function."""

    def test_step_based_completion(self):
        context = MagicMock(spec=TrialContext, num_steps=100)
        # Complete
        complete_result = {"loss_history": [0.1] * 100, "expected_steps": 100}
        assert _is_run_complete(complete_result, context) is True
        # Incomplete
        incomplete_result = {"loss_history": [0.1] * 99, "expected_steps": 100}
        assert _is_run_complete(incomplete_result, context) is False
        # More steps than expected is still complete
        over_result = {"loss_history": [0.1] * 101, "expected_steps": 100}
        assert _is_run_complete(over_result, context) is True

    def test_epoch_based_completion(self):
        context = MagicMock(spec=TrialContext, num_epochs=4)
        # Complete
        complete_result = {"epoch_test_accuracies": [0.9] * 4, "expected_epochs": 4}
        assert _is_run_complete(complete_result, context) is True
        # Incomplete
        incomplete_result = {"epoch_test_accuracies": [0.9] * 3, "expected_epochs": 4}
        assert _is_run_complete(incomplete_result, context) is False

    def test_missing_keys_is_not_complete(self):
        context = MagicMock(spec=TrialContext, num_steps=100)
        assert _is_run_complete({}, context) is False
        assert _is_run_complete({"loss_history": [0.1]}, context) is False  # Missing expected_steps


@patch("batch_size_studies.runner._validate_and_store_partial_result")
@patch("batch_size_studies.runner.get_trial_runner")
@patch("batch_size_studies.runner.RunStatus")
class TestSingleTrialExecution:
    """Unit tests for the `run_single_trial` orchestration function."""

    @pytest.fixture
    def mock_context(self):
        """Provides a fully configured mock TrialContext."""
        context = MagicMock(spec=TrialContext)
        context.run_key = RunKey(32, 0.1)
        context.num_steps = 100
        context.no_save = False
        context.experiment = Mock(spec=ExperimentBase)
        context.checkpoint_manager = Mock()
        context.checkpoint_manager.directory = "/fake/dir"
        return context

    def test_skips_run_if_should_not_run(self, mock_RunStatus, mock_get_runner, mock_validate, mock_context):
        mock_status_instance = mock_RunStatus.return_value
        mock_status_instance.should_run = False

        is_successful = run_single_trial(mock_context, {}, set())

        assert is_successful is True  # Skipped runs are considered "successful" for the sweep
        mock_get_runner.assert_not_called()
        mock_RunStatus.assert_called_once_with(
            mock_context.run_key, {}, set(), mock_context.num_steps, mock_context.no_save
        )

    def test_runs_new_trial_successfully_and_cleans_up(
        self, mock_RunStatus, mock_get_runner, mock_validate, mock_context
    ):
        mock_status_instance = mock_RunStatus.return_value
        mock_status_instance.should_run = True

        mock_runner = mock_get_runner.return_value
        # A result that will be considered complete
        mock_runner.run.return_value = {"loss_history": [0.1] * 100, "expected_steps": 100}

        mock_validate.return_value = True  # Simulate successful validation

        results_dict, failed_runs = {}, set()

        is_successful = run_single_trial(mock_context, results_dict, failed_runs)

        assert is_successful is True
        mock_runner.run.assert_called_once()
        mock_context.checkpoint_manager.cleanup_live_checkpoint.assert_called_once_with(mock_context.run_key)

    def test_incomplete_run_does_not_clean_up(self, mock_RunStatus, mock_get_runner, mock_validate, mock_context):
        mock_status_instance = mock_RunStatus.return_value
        mock_status_instance.should_run = True

        mock_runner = mock_get_runner.return_value
        # An incomplete result
        mock_runner.run.return_value = {"loss_history": [0.1] * 50, "expected_steps": 100}
        mock_validate.return_value = True

        is_successful = run_single_trial(mock_context, {}, set())

        assert is_successful is False
        mock_context.checkpoint_manager.cleanup_live_checkpoint.assert_not_called()

    def test_handles_trial_divergence(self, mock_RunStatus, mock_get_runner, mock_validate, mock_context):
        mock_status_instance = mock_RunStatus.return_value
        mock_status_instance.should_run = True

        mock_runner = mock_get_runner.return_value
        mock_runner.run.return_value = None  # Simulate divergence
        mock_validate.return_value = False

        results_dict, failed_runs = {}, set()

        is_successful = run_single_trial(mock_context, results_dict, failed_runs)

        assert is_successful is False
        mock_runner.run.assert_called_once()
        mock_validate.assert_called_once_with(
            None,
            mock_context.run_key,
            results_dict,
            failed_runs,
            mock_context.experiment,
            mock_context.no_save,
            mock_context.checkpoint_manager.directory,
        )
        mock_context.checkpoint_manager.cleanup_live_checkpoint.assert_not_called()
