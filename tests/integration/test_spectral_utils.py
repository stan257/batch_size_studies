import os
import pickle

import pytest

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import SyntheticExperimentFixedData
from batch_size_studies.spectral_utils import get_spectral_filepath, load_spectral_data


@pytest.fixture
def small_experiment():
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


def test_load_spectral_data_returns_empty_when_missing(tmp_path, small_experiment):
    spectral_dir = tmp_path / "spectral"
    experiments_dir = tmp_path / "experiments"
    result = load_spectral_data(
        small_experiment,
        directory=str(experiments_dir),
        spectral_dir=str(spectral_dir),
    )
    assert result == {}


def test_load_spectral_data_reads_cached_file(tmp_path, small_experiment):
    spectral_dir = tmp_path / "spectral"
    experiments_dir = tmp_path / "experiments"
    run_key = RunKey(batch_size=16, eta=0.1)

    filepath = get_spectral_filepath(
        small_experiment,
        directory=str(experiments_dir),
        spectral_dir=str(spectral_dir),
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cached = {run_key: {100: {"eigenvalues": [1.0, 2.0], "trace": 3.0}}}
    with open(filepath, "wb") as f:
        pickle.dump(cached, f)

    loaded = load_spectral_data(
        small_experiment,
        directory=str(experiments_dir),
        spectral_dir=str(spectral_dir),
    )
    assert loaded[run_key][100]["eigenvalues"] == [1.0, 2.0]
    assert loaded[run_key][100]["trace"] == pytest.approx(3.0)


def test_load_spectral_data_handles_corrupted_file(tmp_path, small_experiment):
    spectral_dir = tmp_path / "spectral"
    experiments_dir = tmp_path / "experiments"
    filepath = get_spectral_filepath(
        small_experiment,
        directory=str(experiments_dir),
        spectral_dir=str(spectral_dir),
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(b"not-a-pickle")

    loaded = load_spectral_data(
        small_experiment,
        directory=str(experiments_dir),
        spectral_dir=str(spectral_dir),
    )
    assert loaded == {}
