import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from batch_size_studies.cli import (
    _handle_list_command,
    _resolve_experiment_configs,
)
from batch_size_studies.definitions import RunKey
from batch_size_studies.experiments import ExperimentBase
from batch_size_studies.runner import (
    RunStatus,
    _all_runs_accounted_for,
    _is_run_result_complete,
    _validate_and_store_partial_result,
)


def test_resolve_experiment_configs_filters_by_name(monkeypatch):
    toy_configs = {"expA": object(), "expB": object()}
    monkeypatch.setattr("batch_size_studies.cli.get_main_experiment_configs", lambda **kwargs: toy_configs)
    args = argparse.Namespace(
        optimizer=None, loss=None, experiment_types=None, name=["expB"], command="list", list_overrides=False
    )
    resolved = _resolve_experiment_configs(args)
    assert list(resolved.keys()) == ["expB"]


def test_resolve_experiment_configs_returns_none_when_name_missing(monkeypatch, caplog):
    monkeypatch.setattr("batch_size_studies.cli.get_main_experiment_configs", lambda **kwargs: {"exp": object()})
    args = argparse.Namespace(
        optimizer=None, loss=None, experiment_types=None, name=["missing"], command="list", list_overrides=False
    )
    with caplog.at_level("ERROR"):
        resolved = _resolve_experiment_configs(args)
    assert resolved is None
    assert "No experiments found with name(s)" in caplog.text


def test_handle_list_command_prints_table(capsys):
    args = argparse.Namespace(list_overrides=False)
    experiment = SimpleNamespace(
        experiment_type="mnist",
        optimizer=SimpleNamespace(name="SGD"),
        loss_type=SimpleNamespace(name="MSE"),
    )
    _handle_list_command(args, {"mnist_exp": experiment})
    out = capsys.readouterr().out
    assert "Available Experiments" in out
    assert "mnist_exp" in out


def test_handle_list_command_shows_overrides(capsys):
    args = argparse.Namespace(list_overrides=True)
    _handle_list_command(args, {})
    out = capsys.readouterr().out
    assert "Supported override keys" in out


class Test_validate_and_store_partial_result:
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


class TestPreFlightHelpers:
    def test_is_run_result_complete_expected_steps(self):
        assert _is_run_result_complete({"expected_steps": 2, "loss_history": [1, 2]})
        assert not _is_run_result_complete({"expected_steps": 2, "loss_history": [1]})

    def test_is_run_result_complete_epoch_fallback(self):
        assert _is_run_result_complete({"epoch_test_accuracies": [0.5]})
        assert not _is_run_result_complete(None)

    def test_all_runs_accounted_for_true(self):
        experiment = Mock()
        experiment.should_skip_batch_size.return_value = False
        results = {RunKey(1, 0.1): {"expected_steps": 1, "loss_history": [0.5]}}

        assert _all_runs_accounted_for(experiment, [1], [0.1], results, set()) is True
        experiment.should_skip_batch_size.assert_called_once_with(1, train_ds=None)

    def test_all_runs_accounted_for_false_when_missing(self):
        experiment = Mock()
        experiment.should_skip_batch_size.return_value = False
        results = {}
        assert _all_runs_accounted_for(experiment, [1], [0.1], results, set()) is False

    def test_all_runs_accounted_for_skips_failed_entries(self):
        experiment = Mock()
        experiment.should_skip_batch_size.return_value = False
        results = {}
        failed = {RunKey(1, 0.1)}
        assert _all_runs_accounted_for(experiment, [1], [0.1], results, failed) is True


def test_run_status_is_successful_requires_completion():
    run_key = RunKey(32, 0.1)
    partial_results = {run_key: {"loss_history": [1.0], "expected_steps": 3}}
    status_partial = RunStatus(run_key, partial_results, set(), num_steps=3, no_save=False)
    assert status_partial.is_successful is False

    complete_results = {run_key: {"loss_history": [1.0, 0.8, 0.6], "expected_steps": 3}}
    status_complete = RunStatus(run_key, complete_results, set(), num_steps=3, no_save=False)
    assert status_complete.is_successful is True
