from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from batch_size_studies.data_utils import (
    extract_loss_histories,
    filter_experiments,
    filter_loss_dict_by_loss_threshold,
    filter_loss_dicts,
    get_loss_history_from_result,
    subsample_loss_dict_periodic,
    uniform_smooth_loss_dicts,
)
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import ExperimentBase, LinearStudentExperiment, MLPStudentExperiment

# Fixtures and mocks


@dataclass(frozen=True)
class MockSynthExperiment(LinearStudentExperiment, ExperimentBase):
    """A mock experiment type for testing default attribute handling."""

    val: int
    experiment_type: str = field(default="synth", init=False)

    def __post_init__(self):
        # The base class __post_init__ does type checking, which we can
        # bypass for this simple mock by not calling super().__post_init__().
        pass

    def is_run_complete(self, result, run_key):
        pass

    def should_skip_batch_size(self, batch_size, train_ds_size=None):
        pass

    def prepare_datasets(self, init_key: int, **kwargs):
        pass

    def get_trial_runner_class(self):
        pass

    def get_model_widths(self) -> list[int]:
        pass

    def get_model_wrapper(self, model_instance, params0):
        pass

    def get_adjusted_eta(self, base_eta: float) -> float:
        pass

    def compute_num_steps(self, batch_size: int, train_ds: Any, num_epochs: int | None) -> tuple[int, int]:
        return 0, 0


@dataclass(frozen=True)
class MockMNISTExperiment(MLPStudentExperiment, ExperimentBase):
    """A mock experiment type with all filterable attributes."""

    val: int
    experiment_type: str = field(default="mnist", init=False)

    def __post_init__(self):
        # Bypassing for simplicity in tests.
        pass

    def is_run_complete(self, result, run_key):
        pass

    def should_skip_batch_size(self, batch_size, train_ds_size=None):
        pass

    def prepare_datasets(self, init_key: int, **kwargs):
        pass

    def get_trial_runner_class(self):
        pass

    def get_model_widths(self) -> list[int]:
        pass

    def get_model_wrapper(self, model_instance, params0):
        pass

    def get_adjusted_eta(self, base_eta: float) -> float:
        pass

    def compute_num_steps(self, batch_size: int, train_ds: Any, num_epochs: int | None) -> tuple[int, int]:
        return 0, 0


@pytest.fixture
def sample_experiments() -> dict[str, ExperimentBase]:
    return {
        "synth_1": MockSynthExperiment(val=1, D=10, optimizer=OptimizerType.SGD, loss_type=LossType.MSE),
        "synth_2": MockSynthExperiment(val=2, D=10, optimizer=OptimizerType.SGD, loss_type=LossType.MSE),
        "mnist_mse_sgd": MockMNISTExperiment(
            val=3,
            loss_type=LossType.MSE,
            optimizer=OptimizerType.SGD,
            parameterization=Parameterization.SP,
            N=16,
            L=2,
            gamma=1.0,
        ),
        "mnist_xent_sgd": MockMNISTExperiment(
            val=4,
            loss_type=LossType.XENT,
            optimizer=OptimizerType.SGD,
            parameterization=Parameterization.SP,
            N=16,
            L=2,
            gamma=1.0,
        ),
        "mnist_xent_adam": MockMNISTExperiment(
            val=5,
            loss_type=LossType.XENT,
            optimizer=OptimizerType.ADAM,
            parameterization=Parameterization.SP,
            N=16,
            L=2,
            gamma=1.0,
        ),
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
    @pytest.mark.parametrize(
        "filter_kwargs, expected_keys",
        [
            ({"experiment_type": MockSynthExperiment}, {"synth_1", "synth_2"}),
            (
                {"experiment_type": MockMNISTExperiment},
                {"mnist_mse_sgd", "mnist_xent_sgd", "mnist_xent_adam"},
            ),
            (
                {"experiment_type": MockMNISTExperiment, "loss_type": LossType.XENT},
                {"mnist_xent_sgd", "mnist_xent_adam"},
            ),
            (
                {"experiment_type": MockMNISTExperiment, "loss_type": LossType.MSE},
                {"mnist_mse_sgd"},
            ),
            (
                {"experiment_type": MockSynthExperiment, "loss_type": LossType.MSE},
                {"synth_1", "synth_2"},
            ),
            ({"experiment_type": MockSynthExperiment, "loss_type": LossType.XENT}, set()),
            (
                {"experiment_type": MockMNISTExperiment, "parameterization": Parameterization.SP},
                {"mnist_mse_sgd", "mnist_xent_sgd", "mnist_xent_adam"},
            ),
            ({"experiment_type": MockMNISTExperiment, "parameterization": Parameterization.MUP}, set()),
            (
                {"experiment_type": MockMNISTExperiment, "optimizer": OptimizerType.SGD},
                {"mnist_mse_sgd", "mnist_xent_sgd"},
            ),
            (
                {"experiment_type": MockMNISTExperiment, "optimizer": OptimizerType.ADAM},
                {"mnist_xent_adam"},
            ),
            (
                {"experiment_type": MockSynthExperiment, "optimizer": OptimizerType.SGD},
                {"synth_1", "synth_2"},
            ),
            ({"experiment_type": MockSynthExperiment, "optimizer": OptimizerType.ADAM}, set()),
            (
                {
                    "experiment_type": MockMNISTExperiment,
                    "loss_type": LossType.XENT,
                    "optimizer": OptimizerType.ADAM,
                },
                {"mnist_xent_adam"},
            ),
            (
                {
                    "experiment_type": MockMNISTExperiment,
                    "loss_type": LossType.MSE,
                    "optimizer": OptimizerType.ADAM,
                },
                set(),
            ),
        ],
        ids=[
            "by_type_synth",
            "by_type_mnist",
            "mnist_by_loss_xent",
            "mnist_by_loss_mse",
            "synth_by_loss_mse",
            "synth_by_loss_xent_no_match",
            "by_param_sp",
            "by_param_mup_no_match",
            "mnist_by_optimizer_sgd",
            "mnist_by_optimizer_adam",
            "synth_by_optimizer_sgd",
            "synth_by_optimizer_adam_no_match",
            "all_criteria_match",
            "all_criteria_no_match",
        ],
    )
    def test_filter_scenarios(self, sample_experiments, filter_kwargs, expected_keys):
        """Tests various filtering scenarios for experiments."""
        filtered = filter_experiments(sample_experiments, **filter_kwargs)
        assert set(filtered.keys()) == expected_keys

        # Verify properties of the filtered items
        for exp in filtered.values():
            for key, value in filter_kwargs.items():
                if key == "experiment_type":
                    assert isinstance(exp, value)
                else:
                    assert getattr(exp, key) == value

    def test_empty_input_dictionary(self):
        """Tests that filtering an empty dictionary results in an empty dictionary."""
        filtered = filter_experiments({}, experiment_type=MockMNISTExperiment)
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


class TestFilterLossDictByLossThreshold:
    @pytest.fixture
    def loss_dict_for_thresholding(self):
        return {
            # Should be kept
            RunKey(16, 0.1): {"loss_history": [1.0, 0.9, 0.8]},
            # Should be kept (loss equals threshold)
            RunKey(16, 0.01): {"loss_history": [1.2, 1.1, 2.0]},
            # Should be removed (one loss > threshold)
            RunKey(32, 0.1): {"loss_history": [0.8, 2.1, 0.6]},
            # Should be removed (contains NaN)
            RunKey(32, 0.01): {"loss_history": [0.9, np.nan, 0.7]},
            # Should be removed (contains inf)
            RunKey(64, 0.1): {"loss_history": [0.5, np.inf, 0.4]},
            # Should be kept (empty history)
            RunKey(64, 0.01): {"loss_history": []},
            # Should be kept (no history key)
            RunKey(128, 0.1): {"other_metric": 123},
        }

    def test_filters_based_on_threshold(self, loss_dict_for_thresholding):
        threshold = 2.0
        filtered_dict = filter_loss_dict_by_loss_threshold(loss_dict_for_thresholding, threshold)

        assert RunKey(16, 0.1) in filtered_dict
        assert RunKey(16, 0.01) in filtered_dict
        assert RunKey(64, 0.01) in filtered_dict
        assert RunKey(128, 0.1) in filtered_dict

        assert RunKey(32, 0.1) not in filtered_dict
        assert RunKey(32, 0.01) not in filtered_dict
        assert RunKey(64, 0.1) not in filtered_dict

        assert len(filtered_dict) == 4

    def test_works_on_list_of_dicts(self, loss_dict_for_thresholding):
        """Tests that the decorator correctly applies the function to a list."""
        threshold = 2.0
        list_of_dicts = [loss_dict_for_thresholding, loss_dict_for_thresholding.copy()]

        filtered_list = filter_loss_dict_by_loss_threshold(list_of_dicts, threshold)

        assert isinstance(filtered_list, list)
        assert len(filtered_list) == 2
        # Check the contents of the first filtered dict
        assert len(filtered_list[0]) == 4
        assert RunKey(32, 0.1) not in filtered_list[0]

    def test_empty_dict_input(self):
        filtered = filter_loss_dict_by_loss_threshold({}, threshold=10.0)
        assert filtered == {}
