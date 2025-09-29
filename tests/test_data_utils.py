from dataclasses import dataclass, field

import numpy as np
import pytest

from batch_size_studies.data_utils import (
    extract_loss_histories,
    filter_experiments,
    filter_loss_dicts,
    get_loss_history_from_result,
    subsample_loss_dict_periodic,
    uniform_smooth_loss_dicts,
)
from batch_size_studies.definitions import LossType, OptimizerType, RunKey
from batch_size_studies.experiments import ExperimentBase

# Fixtures and mocks


@dataclass
class MockSynthExperiment(ExperimentBase):
    """A mock experiment type for testing default attribute handling."""

    val: int
    experiment_type: str = field(default="synth", init=False)

    def __post_init__(self):
        # The base class __post_init__ does type checking, which we can
        # bypass for this simple mock by not calling super().__post_init__().
        pass


@dataclass
class MockMNISTExperiment(ExperimentBase):
    """A mock experiment type with all filterable attributes."""

    val: int
    loss_type: LossType
    optimizer: OptimizerType
    experiment_type: str = field(default="mnist", init=False)

    def __post_init__(self):
        # Bypassing for simplicity in tests.
        pass


@pytest.fixture
def sample_experiments() -> dict[str, ExperimentBase]:
    """Provides a dictionary of various mock experiments for testing."""
    return {
        "synth_1": MockSynthExperiment(val=1),
        "synth_2": MockSynthExperiment(val=2),
        "mnist_mse_sgd": MockMNISTExperiment(val=3, loss_type=LossType.MSE, optimizer=OptimizerType.SGD),
        "mnist_xent_sgd": MockMNISTExperiment(val=4, loss_type=LossType.XENT, optimizer=OptimizerType.SGD),
        "mnist_xent_adam": MockMNISTExperiment(val=5, loss_type=LossType.XENT, optimizer=OptimizerType.ADAM),
    }


@pytest.fixture
def sample_loss_dict():
    """A fixture for a sample loss dictionary."""
    return {
        RunKey(batch_size=16, eta=0.1): [1.0, 0.9, 0.8],
        RunKey(batch_size=16, eta=0.01): [1.2, 1.1, 1.0],
        RunKey(batch_size=32, eta=0.1): [0.8, 0.7, 0.6],
        RunKey(batch_size=32, eta=0.01): [0.9, 0.8, 0.7],
    }


@pytest.fixture
def sample_results_dict():
    """A fixture for a sample results dictionary with the new format."""
    return {
        RunKey(16, 0.1): {"loss_history": [1.0, 0.9], "other_metric": 99},
        RunKey(32, 0.1): {"loss_history": [0.8, 0.7]},
        RunKey(64, 0.1): [0.5, 0.4],  # Mix in old format
        RunKey(128, 0.1): {"other_metric": 123},  # No loss history
    }


@pytest.fixture
def extended_sample_loss_dict():
    """A fixture for a more extensive sample loss dictionary."""
    return {
        RunKey(16, 0.1): [],
        RunKey(16, 0.05): [],
        RunKey(16, 0.025): [],
        RunKey(16, 0.0125): [],
        RunKey(32, 0.1): [],
        RunKey(32, 0.05): [],
        RunKey(32, 0.025): [],
        RunKey(32, 0.0125): [],
        RunKey(64, 0.1): [],
        RunKey(64, 0.05): [],
        RunKey(64, 0.025): [],
        RunKey(64, 0.0125): [],
        RunKey(128, 0.1): [],
        RunKey(128, 0.05): [],
        RunKey(128, 0.025): [],
        RunKey(128, 0.0125): [],
    }


class TestFilterExperiments:
    """Tests for the filter_experiments function."""

    def test_filter_by_experiment_type_only(self, sample_experiments):
        """Tests filtering only by the experiment type."""
        # Filter for synth experiments
        filtered = filter_experiments(sample_experiments, experiment_type=MockSynthExperiment)

        assert len(filtered) == 2
        assert "synth_1" in filtered
        assert "synth_2" in filtered
        assert all(isinstance(exp, MockSynthExperiment) for exp in filtered.values())

        # Filter for MNIST experiments
        filtered_mnist = filter_experiments(sample_experiments, experiment_type=MockMNISTExperiment)

        assert len(filtered_mnist) == 3
        assert "mnist_mse_sgd" in filtered_mnist
        assert "mnist_xent_sgd" in filtered_mnist
        assert "mnist_xent_adam" in filtered_mnist
        assert all(isinstance(exp, MockMNISTExperiment) for exp in filtered_mnist.values())

    def test_filter_by_loss_type(self, sample_experiments):
        """Tests filtering by experiment type and loss type."""
        # Filter for XENT loss
        filtered_xent = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            loss_type=LossType.XENT,
        )

        assert len(filtered_xent) == 2
        assert "mnist_xent_sgd" in filtered_xent
        assert "mnist_xent_adam" in filtered_xent
        assert all(exp.loss_type == LossType.XENT for exp in filtered_xent.values())

        # Filter for MSE loss
        filtered_mse = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            loss_type=LossType.MSE,
        )

        assert len(filtered_mse) == 1
        assert "mnist_mse_sgd" in filtered_mse
        assert filtered_mse["mnist_mse_sgd"].loss_type == LossType.MSE

    def test_filter_by_optimizer(self, sample_experiments):
        """Tests filtering by experiment type and optimizer."""
        # Filter for SGD optimizer
        filtered_sgd = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            optimizer=OptimizerType.SGD,
        )

        assert len(filtered_sgd) == 2
        assert "mnist_mse_sgd" in filtered_sgd
        assert "mnist_xent_sgd" in filtered_sgd
        assert all(exp.optimizer == OptimizerType.SGD for exp in filtered_sgd.values())

        # Filter for Adam optimizer
        filtered_adam = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            optimizer=OptimizerType.ADAM,
        )

        assert len(filtered_adam) == 1
        assert "mnist_xent_adam" in filtered_adam
        assert filtered_adam["mnist_xent_adam"].optimizer == OptimizerType.ADAM

    def test_filter_by_all_criteria(self, sample_experiments):
        """Tests filtering by all available criteria: type, loss, and optimizer."""
        filtered = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            loss_type=LossType.XENT,
            optimizer=OptimizerType.ADAM,
        )

        assert len(filtered) == 1
        assert "mnist_xent_adam" in filtered

        exp = filtered["mnist_xent_adam"]
        assert isinstance(exp, MockMNISTExperiment)
        assert exp.loss_type == LossType.XENT
        assert exp.optimizer == OptimizerType.ADAM

    def test_default_loss_type_handling(self, sample_experiments):
        """Tests that experiments without a `loss_type` attribute are treated as MSE."""
        # MockSynthExperiment does not have a `loss_type` attribute.
        # It should match when filtering for MSE.
        filtered_mse = filter_experiments(
            sample_experiments,
            experiment_type=MockSynthExperiment,
            loss_type=LossType.MSE,
        )

        assert len(filtered_mse) == 2
        assert "synth_1" in filtered_mse
        assert "synth_2" in filtered_mse

        # It should not match when filtering for a different loss type.
        filtered_xent = filter_experiments(
            sample_experiments,
            experiment_type=MockSynthExperiment,
            loss_type=LossType.XENT,
        )

        assert len(filtered_xent) == 0

    def test_missing_optimizer_handling(self, sample_experiments):
        """
        Tests that experiments without an `optimizer` attribute do not match when
        an optimizer is specified in the filter.
        """
        # MockSynthExperiment does not have an `optimizer` attribute.
        # It should not be returned when filtering for any optimizer.
        for optimizer_type in [OptimizerType.SGD, OptimizerType.ADAM]:
            filtered = filter_experiments(
                sample_experiments,
                experiment_type=MockSynthExperiment,
                optimizer=optimizer_type,
            )
            assert len(filtered) == 0

    def test_no_matches(self, sample_experiments):
        """Tests that an empty dictionary is returned when no experiments match."""
        filtered = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            loss_type=LossType.MSE,
            optimizer=OptimizerType.ADAM,  # No experiment has this combination
        )

        assert len(filtered) == 0
        assert filtered == {}

    def test_empty_input_dictionary(self):
        """Tests that an empty dictionary is returned when the input is empty."""
        filtered = filter_experiments({}, experiment_type=MockMNISTExperiment)

        assert len(filtered) == 0
        assert filtered == {}


class TestFilterLossDicts:
    """Tests for the filter_loss_dicts function."""

    def test_filter_by_batch_size(self, sample_loss_dict):
        """Test filtering by batch size."""
        # Remove batch size 32
        filtered = filter_loss_dicts(sample_loss_dict, filter_by="B", values=[32], keep=False)

        assert len(filtered) == 2
        assert RunKey(16, 0.1) in filtered
        assert RunKey(16, 0.01) in filtered
        assert RunKey(32, 0.1) not in filtered
        assert RunKey(32, 0.01) not in filtered

    def test_filter_by_eta(self, sample_loss_dict):
        """Test filtering by learning rate (eta)."""
        # Remove eta 0.01
        filtered = filter_loss_dicts(sample_loss_dict, filter_by="eta", values=[0.01], keep=False)

        assert len(filtered) == 2
        assert RunKey(16, 0.1) in filtered
        assert RunKey(32, 0.1) in filtered
        assert RunKey(16, 0.01) not in filtered
        assert RunKey(32, 0.01) not in filtered

    def test_filter_by_temperature(self):
        """Test filtering by temperature."""
        temp_dict = {
            RunKey(16, 0.16): [],  # temp: 0.01
            RunKey(32, 0.16): [],  # temp: 0.005
        }

        filtered = filter_loss_dicts(temp_dict, filter_by="temp", values=[0.005], keep=False)

        assert len(filtered) == 1
        assert RunKey(16, 0.16) in filtered


class TestSmoothLossDicts:
    """Tests for the uniform_smooth_loss_dicts function."""

    def test_smooth_with_average(self, sample_loss_dict):
        """Test smoothing loss dictionaries with averaging function."""

        def avg_smoother(x):
            return [np.mean(x)]

        smoothed = uniform_smooth_loss_dicts(sample_loss_dict, smoother=avg_smoother)

        assert smoothed[RunKey(16, 0.1)] == pytest.approx([0.9])
        assert smoothed[RunKey(32, 0.1)] == pytest.approx([0.7])


class TestSubsampleLossDict:
    """Tests for the subsample_loss_dict_periodic function."""

    def test_subsample_by_batch_size(self, sample_loss_dict):
        """Test periodic subsampling by batch size."""
        # Add more batch sizes for better testing
        loss_dict = sample_loss_dict.copy()
        loss_dict.update({RunKey(64, 0.1): [0.5], RunKey(128, 0.1): [0.4]})

        # Subsample every 2nd batch size from sorted unique batch sizes [16, 32, 64, 128]
        subsampled = subsample_loss_dict_periodic(loss_dict, subsample_by="batch_size", every=2)

        present_batch_sizes = {k.batch_size for k in subsampled.keys()}
        assert present_batch_sizes == {16, 64}

    def test_subsample_by_both_parameters(self, extended_sample_loss_dict):
        """Tests subsampling by 'both' batch_size and eta."""
        # Unique B: [16, 32, 64, 128]. every=2 -> keep {16, 64}
        # Unique eta: [0.0125, 0.025, 0.05, 0.1]. every=2 -> keep {0.0125, 0.05}
        subsampled = subsample_loss_dict_periodic(extended_sample_loss_dict, subsample_by="both", every=2)

        expected_keys = {
            RunKey(16, 0.0125),
            RunKey(16, 0.05),
            RunKey(64, 0.0125),
            RunKey(64, 0.05),
        }

        assert set(subsampled.keys()) == expected_keys


class TestLossHistoryExtraction:
    """Tests for loss history extraction functions."""

    def test_extract_loss_histories(self):
        """Tests that loss histories are correctly extracted from a results dict."""
        results_dict = {
            RunKey(16, 0.1): {"loss_history": [1.0, 0.9], "other_metric": 99},
            RunKey(32, 0.1): {"loss_history": [0.8, 0.7]},
            RunKey(64, 0.1): {"other_metric": 123},  # No loss history
        }

        histories = extract_loss_histories(results_dict)

        assert len(histories) == 2
        # Use sets to check for presence regardless of order
        assert {tuple(h) for h in histories.values()} == {(1.0, 0.9), (0.8, 0.7)}

    def test_get_loss_history_from_result(self, sample_results_dict):
        """Tests the helper function for extracting loss histories."""
        # New format (dict with 'loss_history' key)
        result_new_format = get_loss_history_from_result(sample_results_dict[RunKey(16, 0.1)])
        assert result_new_format == [1.0, 0.9]

        # Old format (direct list)
        result_old_format = get_loss_history_from_result(sample_results_dict[RunKey(64, 0.1)])
        assert result_old_format == [0.5, 0.4]

        # No history available
        result_no_history = get_loss_history_from_result(sample_results_dict[RunKey(128, 0.1)])
        assert result_no_history is None
