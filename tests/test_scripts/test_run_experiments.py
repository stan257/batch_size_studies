import pytest

from batch_size_studies.definitions import Parameterization, RunKey
from batch_size_studies.experiments import MNIST1MSampledExperiment
from scripts.run_experiments import are_all_runs_complete


# Define a sample experiment configuration for testing
@pytest.fixture
def sample_config():
    return MNIST1MSampledExperiment(
        N=32, L=2, parameterization=Parameterization.MUP, num_epochs=5, max_train_samples=1024
    )


# Define the hyperparameter grid for the test
@pytest.fixture
def sample_grids():
    return [128], [0.1]  # One batch size, one eta


def test_incomplete_epoch_run_is_not_complete(sample_config, sample_grids):
    """
    Verify that a run with fewer epochs than configured is marked as incomplete.
    """
    batch_sizes, etas = sample_grids
    run_key = RunKey(batch_size=128, eta=0.1)

    # Create a results dictionary with a run that only has 2/5 epochs
    incomplete_losses = {run_key: {"epoch_test_accuracies": [0.9, 0.91]}}

    assert not are_all_runs_complete(sample_config, incomplete_losses, batch_sizes, etas)


def test_complete_epoch_run_is_complete(sample_config, sample_grids):
    """
    Verify that a run with the correct number of epochs is marked as complete.
    """
    batch_sizes, etas = sample_grids
    run_key = RunKey(batch_size=128, eta=0.1)

    # Create a results dictionary with a fully completed run
    complete_losses = {run_key: {"epoch_test_accuracies": [0.9, 0.91, 0.92, 0.93, 0.94]}}

    assert are_all_runs_complete(sample_config, complete_losses, batch_sizes, etas)


def test_missing_run_is_not_complete(sample_config, sample_grids):
    """
    Verify that if a run is missing entirely, it's marked as incomplete.
    """
    batch_sizes, etas = sample_grids
    empty_losses = {}

    assert not are_all_runs_complete(sample_config, empty_losses, batch_sizes, etas)
