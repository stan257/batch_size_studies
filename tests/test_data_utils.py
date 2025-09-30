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
    return {
        "synth_1": MockSynthExperiment(val=1),
        "synth_2": MockSynthExperiment(val=2),
        "mnist_mse_sgd": MockMNISTExperiment(val=3, loss_type=LossType.MSE, optimizer=OptimizerType.SGD),
        "mnist_xent_sgd": MockMNISTExperiment(val=4, loss_type=LossType.XENT, optimizer=OptimizerType.SGD),
        "mnist_xent_adam": MockMNISTExperiment(val=5, loss_type=LossType.XENT, optimizer=OptimizerType.ADAM),
    }


@pytest.fixture
def sample_loss_dict():
    return {
        RunKey(batch_size=16, eta=0.1): [1.0, 0.9, 0.8],
        RunKey(batch_size=16, eta=0.01): [1.2, 1.1, 1.0],
        RunKey(batch_size=32, eta=0.1): [0.8, 0.7, 0.6],
        RunKey(batch_size=32, eta=0.01): [0.9, 0.8, 0.7],
    }


@pytest.fixture
def sample_results_dict():
    return {
        RunKey(16, 0.1): {"loss_history": [1.0, 0.9], "other_metric": 99},
        RunKey(32, 0.1): {"loss_history": [0.8, 0.7]},
        RunKey(64, 0.1): [0.5, 0.4],  # Mix in old format
        RunKey(128, 0.1): {"other_metric": 123},  # No loss history
    }


@pytest.fixture
def extended_sample_loss_dict():
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
    def test_filter_by_experiment_type_only(self, sample_experiments):
        filtered = filter_experiments(sample_experiments, experiment_type=MockSynthExperiment)

        assert len(filtered) == 2
        assert "synth_1" in filtered
        assert "synth_2" in filtered
        assert all(isinstance(exp, MockSynthExperiment) for exp in filtered.values())

        filtered_mnist = filter_experiments(sample_experiments, experiment_type=MockMNISTExperiment)

        assert len(filtered_mnist) == 3
        assert "mnist_mse_sgd" in filtered_mnist
        assert "mnist_xent_sgd" in filtered_mnist
        assert "mnist_xent_adam" in filtered_mnist
        assert all(isinstance(exp, MockMNISTExperiment) for exp in filtered_mnist.values())

    def test_filter_by_loss_type(self, sample_experiments):
        filtered_xent = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            loss_type=LossType.XENT,
        )

        assert len(filtered_xent) == 2
        assert "mnist_xent_sgd" in filtered_xent
        assert "mnist_xent_adam" in filtered_xent
        assert all(exp.loss_type == LossType.XENT for exp in filtered_xent.values())

        filtered_mse = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            loss_type=LossType.MSE,
        )

        assert len(filtered_mse) == 1
        assert "mnist_mse_sgd" in filtered_mse
        assert filtered_mse["mnist_mse_sgd"].loss_type == LossType.MSE

    def test_filter_by_optimizer(self, sample_experiments):
        filtered_sgd = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            optimizer=OptimizerType.SGD,
        )

        assert len(filtered_sgd) == 2
        assert "mnist_mse_sgd" in filtered_sgd
        assert "mnist_xent_sgd" in filtered_sgd
        assert all(exp.optimizer == OptimizerType.SGD for exp in filtered_sgd.values())

        filtered_adam = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            optimizer=OptimizerType.ADAM,
        )

        assert len(filtered_adam) == 1
        assert "mnist_xent_adam" in filtered_adam
        assert filtered_adam["mnist_xent_adam"].optimizer == OptimizerType.ADAM

    def test_filter_by_all_criteria(self, sample_experiments):
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
        filtered_mse = filter_experiments(
            sample_experiments,
            experiment_type=MockSynthExperiment,
            loss_type=LossType.MSE,
        )

        assert len(filtered_mse) == 2
        assert "synth_1" in filtered_mse
        assert "synth_2" in filtered_mse

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
        for optimizer_type in [OptimizerType.SGD, OptimizerType.ADAM]:
            filtered = filter_experiments(
                sample_experiments,
                experiment_type=MockSynthExperiment,
                optimizer=optimizer_type,
            )
            assert len(filtered) == 0

    def test_no_matches(self, sample_experiments):
        filtered = filter_experiments(
            sample_experiments,
            experiment_type=MockMNISTExperiment,
            loss_type=LossType.MSE,
            optimizer=OptimizerType.ADAM,
        )

        assert len(filtered) == 0
        assert filtered == {}

    def test_empty_input_dictionary(self):
        filtered = filter_experiments({}, experiment_type=MockMNISTExperiment)

        assert len(filtered) == 0
        assert filtered == {}


class TestFilterLossDicts:
    def test_filter_by_batch_size(self, sample_loss_dict):
        filtered = filter_loss_dicts(sample_loss_dict, filter_by="B", values=[32], keep=False)

        assert len(filtered) == 2
        assert RunKey(16, 0.1) in filtered
        assert RunKey(16, 0.01) in filtered
        assert RunKey(32, 0.1) not in filtered
        assert RunKey(32, 0.01) not in filtered

    def test_filter_by_eta(self, sample_loss_dict):
        filtered = filter_loss_dicts(sample_loss_dict, filter_by="eta", values=[0.01], keep=False)

        assert len(filtered) == 2
        assert RunKey(16, 0.1) in filtered
        assert RunKey(32, 0.1) in filtered
        assert RunKey(16, 0.01) not in filtered
        assert RunKey(32, 0.01) not in filtered

    def test_filter_by_temperature(self):
        temp_dict = {
            RunKey(16, 0.16): [],  # temp: 0.01
            RunKey(32, 0.16): [],  # temp: 0.005
        }

        filtered = filter_loss_dicts(temp_dict, filter_by="temp", values=[0.005], keep=False)

        assert len(filtered) == 1
        assert RunKey(16, 0.16) in filtered


class TestSmoothLossDicts:
    def test_smooth_with_average(self, sample_loss_dict):
        def avg_smoother(x):
            return [np.mean(x)]

        smoothed = uniform_smooth_loss_dicts(sample_loss_dict, smoother=avg_smoother)

        assert smoothed[RunKey(16, 0.1)] == pytest.approx([0.9])
        assert smoothed[RunKey(32, 0.1)] == pytest.approx([0.7])


class TestSubsampleLossDict:
    def test_subsample_by_batch_size(self, sample_loss_dict):
        loss_dict = sample_loss_dict.copy()
        loss_dict.update({RunKey(64, 0.1): [0.5], RunKey(128, 0.1): [0.4]})

        subsampled = subsample_loss_dict_periodic(loss_dict, subsample_by="batch_size", every=2)

        present_batch_sizes = {k.batch_size for k in subsampled.keys()}
        assert present_batch_sizes == {16, 64}

    def test_subsample_by_both_parameters(self, extended_sample_loss_dict):
        subsampled = subsample_loss_dict_periodic(extended_sample_loss_dict, subsample_by="both", every=2)

        expected_keys = {
            RunKey(16, 0.0125),
            RunKey(16, 0.05),
            RunKey(64, 0.0125),
            RunKey(64, 0.05),
        }

        assert set(subsampled.keys()) == expected_keys


class TestLossHistoryExtraction:
    def test_extract_loss_histories(self):
        results_dict = {
            RunKey(16, 0.1): {"loss_history": [1.0, 0.9], "other_metric": 99},
            RunKey(32, 0.1): {"loss_history": [0.8, 0.7]},
            RunKey(64, 0.1): {"other_metric": 123},  # No loss history
        }

        histories = extract_loss_histories(results_dict)

        assert len(histories) == 2
        assert {tuple(h) for h in histories.values()} == {(1.0, 0.9), (0.8, 0.7)}

    def test_get_loss_history_from_result(self, sample_results_dict):
        result_new_format = get_loss_history_from_result(sample_results_dict[RunKey(16, 0.1)])
        assert result_new_format == [1.0, 0.9]

        result_old_format = get_loss_history_from_result(sample_results_dict[RunKey(64, 0.1)])
        assert result_old_format == [0.5, 0.4]

        result_no_history = get_loss_history_from_result(sample_results_dict[RunKey(128, 0.1)])
        assert result_no_history is None
