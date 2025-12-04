import importlib.util
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import batch_size_studies.spectral.pipeline as spectral_service_module
import batch_size_studies.spectral.spectral_utils as spectral_utils_module
from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import SyntheticExperimentFixedData

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "gather_hessian_spectra.py"
spec = importlib.util.spec_from_file_location("gather_hessian_spectra", SCRIPT_PATH)
spectra_cli = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(spectra_cli)


def _make_experiment():
    return SyntheticExperimentFixedData(
        D=4,
        P=32,
        N=4,
        K=2,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
        seed=0,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
        num_epochs=1,
    )


def _write_weights_file(experiment, experiments_dir, run_key, snapshots=None):
    manager = CheckpointManager(experiment, directory=experiments_dir)
    os.makedirs(os.path.dirname(manager.weights_filepath), exist_ok=True)
    payload = {"initial_params": {"w": np.zeros(1)}, "weight_snapshots": {}}
    if snapshots is not None:
        payload["weight_snapshots"][run_key] = snapshots
    with open(manager.weights_filepath, "wb") as f:
        pickle.dump(payload, f)
    return manager


def _make_args(experiments_dir, **overrides):
    defaults = {
        "experiment": "dummy_experiment",
        "batch_size": 16,
        "eta": 0.1,
        "steps": None,
        "list_only": False,
        "dry_run": False,
        "experiments_dir": experiments_dir,
        "force_recompute": False,
        "num_eigenvalues": 3,
        "num_hessian_samples": 16,
        "hessian_batch_size": 4,
        "max_iter": 5,
        "eig_tol": 1e-3,
        "trace_samples": 10,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class DummyHessianComputer:
    def __init__(self, eigenvalues, trace_value):
        self._eigs = np.asarray(eigenvalues, dtype=float)
        self._trace = float(trace_value)

    def eigenvalues(self, params, key, max_iter, tol, top_n):
        return self._eigs[:top_n], None

    def trace(self, params, key, max_iter):
        return np.array(self._trace), None


class StubbedEvaluator:
    """Deterministic HessianEvaluator replacement returning preset eigen data."""

    def __init__(self, eigenvalues, trace_value):
        self.params = None
        self.key = None
        self.hessian_computer = DummyHessianComputer(eigenvalues, trace_value)


def _patch_hessian_evaluator(monkeypatch, mapping):
    """Map steps to eigen/trace tuples."""

    def _factory(experiment, run_key, step, directory, **kwargs):
        eigen, trace_val = mapping[step]
        return StubbedEvaluator(eigen, trace_val)

    monkeypatch.setattr(spectral_service_module, "HessianEvaluator", _factory)


@pytest.fixture
def experiment_setup(tmp_path, monkeypatch):
    experiment = _make_experiment()
    run_key = RunKey(batch_size=16, eta=0.1)
    experiments_dir = tmp_path / "experiments"
    spectral_dir = tmp_path / "spectral"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(spectra_cli, "SPECTRAL_DATA_DIR", str(spectral_dir))
    monkeypatch.setattr(spectral_utils_module, "SPECTRAL_DATA_DIR", str(spectral_dir))
    monkeypatch.setattr(spectra_cli, "_load_experiment", lambda _: experiment)
    return SimpleNamespace(
        experiment=experiment,
        run_key=run_key,
        experiments_dir=str(experiments_dir),
        spectral_dir=str(spectral_dir),
    )


def test_compute_spectrum_errors_when_snapshots_missing(experiment_setup, caplog):
    args = _make_args(experiment_setup.experiments_dir)
    caplog.set_level("ERROR")
    spectra_cli.compute_spectrum(args)

    assert any("No snapshots found" in record.message for record in caplog.records)
    spectra_path = spectral_utils_module.get_spectral_filepath(
        experiment_setup.experiment,
        directory=experiment_setup.experiments_dir,
        spectral_dir=experiment_setup.spectral_dir,
    )
    assert not os.path.exists(spectra_path)


def test_list_only_reports_steps_without_creating_output(experiment_setup, caplog):
    snapshots = {10: {"layer": np.array([0.1])}, 30: {"layer": np.array([0.2])}}
    _write_weights_file(
        experiment_setup.experiment,
        experiment_setup.experiments_dir,
        experiment_setup.run_key,
        snapshots=snapshots,
    )
    caplog.set_level("INFO")
    args = _make_args(experiment_setup.experiments_dir, list_only=True)
    spectra_cli.compute_spectrum(args)

    assert any("Snapshot steps" in record.message for record in caplog.records)
    spectra_path = spectral_utils_module.get_spectral_filepath(
        experiment_setup.experiment,
        directory=experiment_setup.experiments_dir,
        spectral_dir=experiment_setup.spectral_dir,
    )
    assert not os.path.exists(spectra_path)


def test_dry_run_reports_missing_steps_without_writing(experiment_setup, caplog):
    snapshots = {
        5: {"layer": np.array([0.1])},
        10: {"layer": np.array([0.2])},
    }
    _write_weights_file(
        experiment_setup.experiment,
        experiment_setup.experiments_dir,
        experiment_setup.run_key,
        snapshots=snapshots,
    )
    spectra_path = spectral_utils_module.get_spectral_filepath(
        experiment_setup.experiment,
        directory=experiment_setup.experiments_dir,
        spectral_dir=experiment_setup.spectral_dir,
    )
    cached = {experiment_setup.run_key: {5: {"eigenvalues": [1.0, 2.0, 3.0], "trace": 6.0}}}
    os.makedirs(os.path.dirname(spectra_path), exist_ok=True)
    with open(spectra_path, "wb") as f:
        pickle.dump(cached, f)

    caplog.set_level("INFO")
    args = _make_args(experiment_setup.experiments_dir, steps=[5, 10], dry_run=True)
    spectra_cli.compute_spectrum(args)

    assert any("Dry-run" in record.message for record in caplog.records)
    with open(spectra_path, "rb") as f:
        assert pickle.load(f) == cached


def test_cache_is_reused_and_force_recompute_updates(experiment_setup, monkeypatch):
    _write_weights_file(
        experiment_setup.experiment,
        experiment_setup.experiments_dir,
        experiment_setup.run_key,
        snapshots={100: {"layer": np.array([0.1])}},
    )
    spectra_path = spectral_utils_module.get_spectral_filepath(
        experiment_setup.experiment,
        directory=experiment_setup.experiments_dir,
        spectral_dir=experiment_setup.spectral_dir,
    )
    cached = {experiment_setup.run_key: {100: {"eigenvalues": [1.0, 2.0, 3.0], "trace": 6.0}}}
    os.makedirs(os.path.dirname(spectra_path), exist_ok=True)
    with open(spectra_path, "wb") as f:
        pickle.dump(cached, f)

    args = _make_args(experiment_setup.experiments_dir, steps=[100], num_eigenvalues=2)
    spectra_cli.compute_spectrum(args)
    with open(spectra_path, "rb") as f:
        assert pickle.load(f) == cached

    _patch_hessian_evaluator(monkeypatch, {100: ([9.0, 8.0, 7.0, 6.0], 42.0)})
    args_force = _make_args(
        experiment_setup.experiments_dir,
        steps=[100],
        num_eigenvalues=4,
        force_recompute=True,
    )
    spectra_cli.compute_spectrum(args_force)
    with open(spectra_path, "rb") as f:
        refreshed = pickle.load(f)

    assert refreshed[experiment_setup.run_key][100]["eigenvalues"] == [9.0, 8.0, 7.0, 6.0]
    assert refreshed[experiment_setup.run_key][100]["trace"] == pytest.approx(42.0)


def test_multi_step_serialization_records_each_step(experiment_setup, monkeypatch):
    snapshots = {
        5: {"layer": np.array([0.1])},
        10: {"layer": np.array([0.2])},
    }
    _write_weights_file(
        experiment_setup.experiment,
        experiment_setup.experiments_dir,
        experiment_setup.run_key,
        snapshots=snapshots,
    )
    _patch_hessian_evaluator(
        monkeypatch,
        {
            5: ([1.0, 1.5], 2.5),
            10: ([3.0, 4.0], 6.5),
        },
    )

    args = _make_args(experiment_setup.experiments_dir, num_eigenvalues=2)
    spectra_cli.compute_spectrum(args)

    spectra_path = spectral_utils_module.get_spectral_filepath(
        experiment_setup.experiment,
        directory=experiment_setup.experiments_dir,
        spectral_dir=experiment_setup.spectral_dir,
    )
    with open(spectra_path, "rb") as f:
        stored = pickle.load(f)

    assert set(stored[experiment_setup.run_key].keys()) == {5, 10}
    assert stored[experiment_setup.run_key][5]["eigenvalues"] == [1.0, 1.5]
    assert stored[experiment_setup.run_key][10]["trace"] == pytest.approx(6.5)


def test_compute_spectrum_errors_when_requesting_missing_step(experiment_setup):
    _write_weights_file(
        experiment_setup.experiment,
        experiment_setup.experiments_dir,
        experiment_setup.run_key,
        snapshots={10: {"layer": np.array([0.1])}},
    )
    args = _make_args(experiment_setup.experiments_dir, steps=[10, 40])
    with pytest.raises(ValueError, match="Requested steps"):
        spectra_cli.compute_spectrum(args)

    spectra_path = spectral_utils_module.get_spectral_filepath(
        experiment_setup.experiment,
        directory=experiment_setup.experiments_dir,
        spectral_dir=experiment_setup.spectral_dir,
    )
    assert not os.path.exists(spectra_path)


def test_cache_skip_logs_message(experiment_setup, caplog):
    snapshots = {100: {"layer": np.array([0.3])}}
    _write_weights_file(
        experiment_setup.experiment,
        experiment_setup.experiments_dir,
        experiment_setup.run_key,
        snapshots=snapshots,
    )
    spectra_path = spectral_utils_module.get_spectral_filepath(
        experiment_setup.experiment,
        directory=experiment_setup.experiments_dir,
        spectral_dir=experiment_setup.spectral_dir,
    )
    cached = {experiment_setup.run_key: {100: {"eigenvalues": [1.0, 2.0, 3.0], "trace": 6.0}}}
    os.makedirs(os.path.dirname(spectra_path), exist_ok=True)
    with open(spectra_path, "wb") as f:
        pickle.dump(cached, f)

    caplog.set_level("INFO")
    args = _make_args(experiment_setup.experiments_dir, steps=[100], num_eigenvalues=2)
    spectra_cli.compute_spectrum(args)

    assert any("Existing entry already has" in record.message for record in caplog.records)
    with open(spectra_path, "rb") as f:
        assert pickle.load(f) == cached


def test_partial_step_selection_does_not_touch_other_steps(experiment_setup, monkeypatch):
    snapshots = {
        5: {"layer": np.array([0.1])},
        10: {"layer": np.array([0.2])},
    }
    _write_weights_file(
        experiment_setup.experiment,
        experiment_setup.experiments_dir,
        experiment_setup.run_key,
        snapshots=snapshots,
    )
    spectra_path = spectral_utils_module.get_spectral_filepath(
        experiment_setup.experiment,
        directory=experiment_setup.experiments_dir,
        spectral_dir=experiment_setup.spectral_dir,
    )
    os.makedirs(os.path.dirname(spectra_path), exist_ok=True)
    cached = {experiment_setup.run_key: {5: {"eigenvalues": [3.0], "trace": 1.0}}}
    with open(spectra_path, "wb") as f:
        pickle.dump(cached, f)

    _patch_hessian_evaluator(
        monkeypatch,
        {
            10: ([4.0, 5.0], 7.0),
        },
    )
    args = _make_args(experiment_setup.experiments_dir, steps=[10], num_eigenvalues=2)
    spectra_cli.compute_spectrum(args)

    with open(spectra_path, "rb") as f:
        stored = pickle.load(f)

    assert stored[experiment_setup.run_key][5]["eigenvalues"] == [3.0]
    assert stored[experiment_setup.run_key][10]["eigenvalues"] == [4.0, 5.0]
