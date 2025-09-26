import numpy as np
import pytest

from batch_size_studies.data_utils import (
    extract_loss_histories,
    filter_loss_dicts,
    get_loss_history_from_result,
    subsample_loss_dict_periodic,
    uniform_smooth_loss_dicts,
)
from batch_size_studies.definitions import RunKey


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


class TestDataUtils:
    def test_filter_loss_dicts(self, sample_loss_dict):
        # Filter by batch size
        filtered = filter_loss_dicts(sample_loss_dict, filter_by="B", values=[32], keep=False)
        assert len(filtered) == 2
        assert RunKey(16, 0.1) in filtered
        assert RunKey(32, 0.1) not in filtered

        # Filter by eta
        filtered = filter_loss_dicts(sample_loss_dict, filter_by="eta", values=[0.01], keep=False)
        assert len(filtered) == 2
        assert RunKey(16, 0.1) in filtered
        assert RunKey(32, 0.1) in filtered

        # Filter by temp
        temp_dict = {RunKey(16, 0.16): [], RunKey(32, 0.16): []}  # temps: 0.01, 0.005
        filtered = filter_loss_dicts(temp_dict, filter_by="temp", values=[0.005], keep=False)
        assert len(filtered) == 1
        assert RunKey(16, 0.16) in filtered

    def test_smooth_loss_dicts(self, sample_loss_dict):
        def avg_smoother(x):
            return [np.mean(x)]

        smoothed = uniform_smooth_loss_dicts(sample_loss_dict, smoother=avg_smoother)
        assert smoothed[RunKey(16, 0.1)] == pytest.approx([0.9])
        assert smoothed[RunKey(32, 0.1)] == pytest.approx([0.7])

    def test_subsample_loss_dict_periodic(self, sample_loss_dict):
        loss_dict = sample_loss_dict.copy()
        loss_dict.update({RunKey(64, 0.1): [0.5], RunKey(128, 0.1): [0.4]})

        # Subsample every 2nd batch size from sorted unique batch sizes [16, 32, 64, 128]
        subsampled = subsample_loss_dict_periodic(loss_dict, subsample_by="batch_size", every=2)

        present_bs = {k.batch_size for k in subsampled.keys()}
        assert present_bs == {16, 64}

    def test_subsample_loss_dict_periodic_both(self, extended_sample_loss_dict):
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

    def test_extract_loss_histories(self):
        """Tests that loss histories are correctly extracted from a results dict."""
        results_dict = {
            RunKey(16, 0.1): {"loss_history": [1.0, 0.9], "other_metric": 99},
            RunKey(32, 0.1): {"loss_history": [0.8, 0.7]},
            # A run with no loss history
            RunKey(64, 0.1): {"other_metric": 123},
        }

        histories = extract_loss_histories(results_dict)

        assert len(histories) == 2
        # Use sets to check for presence regardless of order
        assert {tuple(h) for h in histories.values()} == {(1.0, 0.9), (0.8, 0.7)}

    def test_get_loss_history_from_result(self, sample_results_dict):
        """Tests the helper function for extracting loss histories."""
        # New format
        res1 = get_loss_history_from_result(sample_results_dict[RunKey(16, 0.1)])
        assert res1 == [1.0, 0.9]
        # Old format
        res2 = get_loss_history_from_result(sample_results_dict[RunKey(64, 0.1)])
        assert res2 == [0.5, 0.4]
        # No history
        res3 = get_loss_history_from_result(sample_results_dict[RunKey(128, 0.1)])
        assert res3 is None
