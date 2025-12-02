import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import ExperimentBase, SyntheticExperimentFixedTime
from batch_size_studies.runner import (
    TrialContext,
    _all_runs_accounted_for,
    _handle_list_command,
    _handle_run_command,
    _is_run_result_complete,
    _resolve_experiment_configs,
    _validate_and_store_partial_result,
    run_experiment_sweep,
    run_from_cli_args,
    run_single_trial,
)


def test_resolve_experiment_configs_filters_by_name(monkeypatch):
    toy_configs = {"expA": object(), "expB": object()}
    monkeypatch.setattr("batch_size_studies.runner.get_main_experiment_configs", lambda **kwargs: toy_configs)
    args = argparse.Namespace(
        optimizer=None, loss=None, experiment_types=None, name=["expB"], command="list", list_overrides=False
    )
    resolved = _resolve_experiment_configs(args)
    assert list(resolved.keys()) == ["expB"]


def test_resolve_experiment_configs_returns_none_when_name_missing(monkeypatch, caplog):
    monkeypatch.setattr("batch_size_studies.runner.get_main_experiment_configs", lambda **kwargs: {"exp": object()})
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


def test_handle_run_command_dry_run(monkeypatch):
    monkeypatch.setattr("batch_size_studies.runner.get_main_hyperparameter_grids", lambda: ([8, 16], [0.1, 0.2]))
    sweep_calls = []

    def fake_sweep(**kwargs):
        sweep_calls.append(kwargs)

    monkeypatch.setattr("batch_size_studies.runner.run_experiment_sweep", fake_sweep)
    args = argparse.Namespace(
        dry_run=True,
        dry_run_steps=7,
        max_eval_samples=123,
        override=None,
        no_save=True,
        eta_stability_depth=None,
        num_processes=1,
        save_interstitial_snapshots=None,
        save_epoch_snapshots=None,
    )
    experiments = {"demo": object()}
    _handle_run_command(args, experiments)
    assert len(sweep_calls) == 1
    call_kwargs = sweep_calls[0]
    assert call_kwargs["dry_run"] is True
    assert call_kwargs["dry_run_steps"] == 7
    assert call_kwargs["max_eval_samples"] == 123


def test_run_from_args_orchestration(monkeypatch):
    """
    Tests that run_from_cli_args correctly parses args, filters experiments,
    and dispatches to the runner logic. This test is moved from the old
    test_cli.py to directly test the core logic instead of the CLI layer.
    """
    # 1. Setup Mocks and Fakes
    batch_sizes = [4]
    etas = [0.1]
    toy_experiment = SyntheticExperimentFixedTime(
        D=2,
        P=8,
        N=4,
        K=2,
        num_steps=3,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )

    monkeypatch.setattr(
        "batch_size_studies.runner.get_main_experiment_configs", lambda **kwargs: {"toy": toy_experiment}
    )
    monkeypatch.setattr("batch_size_studies.runner.get_main_hyperparameter_grids", lambda: (batch_sizes, etas))

    recorded_calls = []

    def fake_run_single(*args, **kwargs):
        # Corresponds to _run_single_experiment in runner.py
        # def _run_single_experiment(name, experiment_config, batch_sizes, etas, directory, no_save, ...)
        no_save_arg = args[5]
        recorded_calls.append((args[0], no_save_arg))
        return args[0]

    monkeypatch.setattr("batch_size_studies.runner._run_single_experiment", fake_run_single)

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class DummyExecutor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, func, *args, **kwargs):
            return DummyFuture(func(*args, **kwargs))

    monkeypatch.setattr("batch_size_studies.runner.ProcessPoolExecutor", lambda max_workers: DummyExecutor())
    monkeypatch.setattr("batch_size_studies.runner.as_completed", lambda futures: futures)

    # 2. Setup Arguments
    args = argparse.Namespace(
        command="run",
        name=["toy"],
        no_save=True,
        # Set defaults for other args
        optimizer=None,
        loss=None,
        experiment_types=None,
        override=None,
        eta_stability_depth=None,
        max_eval_samples=None,
        num_processes=1,
        save_interstitial_snapshots=None,
        save_epoch_snapshots=None,
    )

    # 3. Run and Assert
    run_from_cli_args(args)
    assert recorded_calls == [("toy", True)]


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


@patch("batch_size_studies.runner._validate_and_store_partial_result")
@patch("batch_size_studies.runner.get_trial_runner")
@patch("batch_size_studies.runner.RunStatus")
class TestSingleTrialExecution:
    @pytest.fixture
    def mock_context(self):
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
        mock_runner.is_complete.return_value = True

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
        mock_runner.is_complete.return_value = False
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


@patch("batch_size_studies.runner._execute_sweep_loops")
@patch("batch_size_studies.runner._setup_sweep_state")
@patch("batch_size_studies.runner._all_runs_accounted_for")
def test_run_experiment_sweep_skips_when_preflight_satisfied(mock_all_accounted, mock_setup_state, mock_execute):
    mock_all_accounted.return_value = True
    checkpoint_manager = Mock()
    checkpoint_manager.directory = "/tmp"
    mock_setup_state.return_value = ({}, set(), checkpoint_manager, object(), object())

    experiment = Mock()
    experiment.prepare_datasets = Mock()

    results, failed = run_experiment_sweep(experiment, batch_sizes=[1], etas=[0.1])

    assert results == {}
    assert failed == set()
    experiment.prepare_datasets.assert_not_called()
    mock_execute.assert_not_called()
