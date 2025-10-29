import pytest

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import MNIST1MSampledExperiment
from scripts.run_experiments import are_all_runs_complete


@pytest.fixture
def sample_config():
    return MNIST1MSampledExperiment(
        N=32,
        L=2,
        parameterization=Parameterization.MUP,
        num_epochs=5,
        max_train_samples=1024,
        gamma=1.0,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )


@pytest.fixture
def sample_grids():
    return [128], [0.1]


def test_incomplete_epoch_run_is_not_complete(sample_config, sample_grids):
    batch_sizes, etas = sample_grids
    run_key = RunKey(batch_size=128, eta=0.1)

    # Create a results dictionary with a run that only has 2/5 epochs
    incomplete_losses = {run_key: {"epoch_test_accuracies": [0.9, 0.91]}}
    failed_runs = set()

    assert not are_all_runs_complete(sample_config, incomplete_losses, failed_runs, batch_sizes, etas)


def test_complete_epoch_run_is_complete(sample_config, sample_grids):
    batch_sizes, etas = sample_grids
    run_key = RunKey(batch_size=128, eta=0.1)
    failed_runs = set()

    # Create a results dictionary with a fully completed run
    complete_losses = {run_key: {"epoch_test_accuracies": [0.9, 0.91, 0.92, 0.93, 0.94]}}

    assert are_all_runs_complete(sample_config, complete_losses, failed_runs, batch_sizes, etas)


def test_missing_run_is_not_complete(sample_config, sample_grids):
    batch_sizes, etas = sample_grids
    empty_losses = {}
    failed_runs = set()

    assert not are_all_runs_complete(sample_config, empty_losses, failed_runs, batch_sizes, etas)


def test_all_runs_failed_is_complete(sample_config, sample_grids):
    """Tests that an experiment is considered complete if all runs have failed."""
    batch_sizes, etas = sample_grids
    run_key = RunKey(batch_size=128, eta=0.1)
    empty_losses = {}
    failed_runs = {run_key}

    assert are_all_runs_complete(sample_config, empty_losses, failed_runs, batch_sizes, etas)


def test_mixed_complete_and_failed_is_complete(sample_config):
    batch_sizes = [128, 256]
    etas = [0.1, 0.01]

    complete_run = RunKey(batch_size=128, eta=0.1)
    failed_runs_set = {RunKey(128, 0.01), RunKey(256, 0.1), RunKey(256, 0.01)}

    losses = {complete_run: {"epoch_test_accuracies": [0.9] * 5}}

    assert are_all_runs_complete(sample_config, losses, failed_runs_set, batch_sizes, etas)


def test_skipped_batch_size_is_complete(sample_config):
    # max_train_samples is 1024, so batch size 2048 should be skipped.
    batch_sizes = [128, 2048]
    etas = [0.1]

    # The results only contain the valid batch size run.
    losses = {RunKey(128, 0.1): {"epoch_test_accuracies": [0.9] * 5}}
    failed_runs = set()

    assert are_all_runs_complete(sample_config, losses, failed_runs, batch_sizes, etas)
