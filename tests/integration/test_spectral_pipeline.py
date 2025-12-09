import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import SyntheticExperimentFixedData
from batch_size_studies.spectral import pipeline as spectral_pipeline
from batch_size_studies.spectral.spectral_utils import get_spectral_filepath


def _make_experiment():
    return SyntheticExperimentFixedData(
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
        D=4,
        P=32,
        K=2,
        N=4,
        L=2,
        parameterization=Parameterization.SP,
        gamma=1.0,
        num_epochs=1,
        seed=0,
    )


def _write_weights_file(experiment, experiments_dir, run_key, snapshots):
    from batch_size_studies.checkpoint_utils import CheckpointManager

    manager = CheckpointManager(experiment, directory=experiments_dir)
    payload = {
        "initial_params": {"layer": np.zeros(1)},
        "weight_snapshots": {run_key: snapshots},
        "metadata": {"init_key": 0},
    }
    os.makedirs(os.path.dirname(manager.weights_filepath), exist_ok=True)
    with open(manager.weights_filepath, "wb") as f:
        pickle.dump(payload, f)


class StubbedEvaluator:
    def __init__(self, eigenvalues, trace_value):
        self._eigs = eigenvalues
        self._trace = trace_value

    def top_eigenvalues(self, top_n, max_iter, tol):
        return self._eigs[:top_n], None

    def trace(self, max_iter):
        return self._trace


def _patch_hessian_evaluator(monkeypatch, mapping):
    def _factory(*args, step, **kwargs):
        eigen, trace_val = mapping[step]
        return StubbedEvaluator(eigen, trace_val)

    monkeypatch.setattr(spectral_pipeline, "HessianEvaluator", _factory)


@pytest.fixture
def experiment_setup(tmp_path):
    experiment = _make_experiment()
    run_key = RunKey(batch_size=16, eta=0.1)
    experiments_dir = tmp_path / "experiments"
    spectral_dir = tmp_path / "spectral"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        experiment=experiment,
        run_key=run_key,
        experiments_dir=str(experiments_dir),
        spectral_dir=str(spectral_dir),
    )


def test_list_snapshot_steps_returns_sorted(experiment_setup):
    _write_weights_file(
        experiment_setup.experiment,
        experiment_setup.experiments_dir,
        experiment_setup.run_key,
        snapshots={20: {"layer": np.array([0.1])}, 5: {"layer": np.array([0.2])}},
    )
    steps = spectral_pipeline.list_snapshot_steps(
        experiment_setup.experiment,
        experiment_setup.run_key,
        experiment_setup.experiments_dir,
    )
    assert steps == [5, 20]


def test_gather_spectra_writes_entries(monkeypatch, experiment_setup):
    steps = {5: {"layer": np.array([0.1])}, 10: {"layer": np.array([0.2])}}
    _write_weights_file(
        experiment_setup.experiment,
        experiment_setup.experiments_dir,
        experiment_setup.run_key,
        snapshots=steps,
    )
    _patch_hessian_evaluator(monkeypatch, {5: ([1.0, 1.5], 2.0), 10: ([3.0], 5.0)})

    spectral_path = get_spectral_filepath(
        experiment_setup.experiment,
        directory=experiment_setup.experiments_dir,
        spectral_dir=experiment_setup.spectral_dir,
    )
    spectral_pipeline.gather_spectra(
        experiment_setup.experiment,
        experiment_setup.run_key,
        [5, 10],
        directory=experiment_setup.experiments_dir,
        spectral_dir=experiment_setup.spectral_dir,
        num_eigenvalues=2,
        num_hessian_samples=32,
        hessian_batch_size=16,
        max_iter=10,
        eig_tol=1e-3,
        trace_samples=5,
    )
    with open(spectral_path, "rb") as f:
        stored = pickle.load(f)

    assert stored[experiment_setup.run_key][5]["eigenvalues"] == [1.0, 1.5]
    assert stored[experiment_setup.run_key][10]["trace"] == 5.0
