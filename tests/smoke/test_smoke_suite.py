import subprocess
import sys

import pytest

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization
from batch_size_studies.experiments import SyntheticExperimentFixedTime
from batch_size_studies.runner import run_experiment_sweep

pytestmark = pytest.mark.smoke


def test_run_experiments_cli_list_smoke():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "list",
            "--optimizer",
            "sgd",
            "--loss",
            "mse",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "mnist1m_mup_MSE_SGD" in result.stdout


def test_tiny_synthetic_sweep(tmp_path):
    config = SyntheticExperimentFixedTime(
        D=2,
        P=8,
        N=4,
        K=2,
        num_steps=5,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )
    results, failures = run_experiment_sweep(
        experiment=config,
        batch_sizes=[2],
        etas=[0.01],
        init_key=0,
        directory=tmp_path,
    )
    assert not failures
    assert results


def test_gather_hessian_list_only(tmp_path):
    cmd = [
        sys.executable,
        "scripts/gather_hessian_spectra.py",
        "--experiment",
        "mnist1m_mup_MSE_SGD_gamma1p0_epochs1",
        "--batch-size",
        "16",
        "--eta",
        "2.0",
        "--list-only",
        "--experiments-dir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0
