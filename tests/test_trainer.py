from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from batch_size_studies.definitions import LossType, OptimizerType, RunKey
from batch_size_studies.trainer import (
    MNISTTrialRunner,
    SyntheticFixedDataTrialRunner,
    SyntheticFixedTimeTrialRunner,
)


class TestMNISTTrialRunnerUnit:
    """Unit tests for the MNISTTrialRunner class."""

    @pytest.fixture
    def mnist_runner(self):
        # Create a mock context object with necessary attributes
        context = MagicMock()
        context.run_key = RunKey(batch_size=64, eta=0.1)
        context.num_epochs = 5
        context.train_ds = {"image": np.zeros((1280, 784))}  # 1280 samples
        context.test_ds = {"image": np.zeros((100, 784)), "label": np.zeros(100)}
        context.experiment = MagicMock()
        context.experiment.D = 784
        context.experiment.optimizer = OptimizerType.SGD
        context.experiment.loss_type = LossType.XENT

        # Mock methods that would be called
        runner = MNISTTrialRunner(context)
        runner.eval_step = Mock(return_value=0.95)  # Mock the JITted eval step
        runner.pbar = Mock()  # Mock the progress bar
        return runner

    def test_init_calculates_steps_per_epoch(self, mnist_runner):
        # 1280 samples / 64 batch_size = 20 steps_per_epoch
        assert mnist_runner.steps_per_epoch == 20

    def test_should_save_checkpoint(self, mnist_runner):
        # Should save at the end of an epoch
        assert mnist_runner._should_save_checkpoint(step=19) is True  # step 19 is the 20th step
        assert mnist_runner._should_save_checkpoint(step=39) is True
        # Should not save in the middle of an epoch
        assert mnist_runner._should_save_checkpoint(step=18) is False
        assert mnist_runner._should_save_checkpoint(step=0) is False

    def test_post_epoch_hook(self, mnist_runner):
        params = "dummy_params"
        results = {"epoch_test_accuracies": []}

        # EVAL_BATCH_SIZE is 512. test_ds has 100 samples. So one loop.
        updated_results = mnist_runner._post_epoch_hook(epoch=0, params=params, results=results)

        # Check that eval_step was called once
        assert mnist_runner.eval_step.call_count == 1
        # Check that the accuracy was appended
        assert len(updated_results["epoch_test_accuracies"]) == 1
        assert updated_results["epoch_test_accuracies"][0] == pytest.approx(0.95)
        # Check that pbar was updated
        mnist_runner.pbar.set_postfix.assert_called_once_with(accuracy="0.9500")

    def test_post_training_hook(self, mnist_runner):
        results = {"epoch_test_accuracies": [0.8, 0.9, 0.95]}
        updated_results = mnist_runner._post_training_hook(params="dummy", results=results)
        assert "final_test_accuracy" in updated_results
        assert updated_results["final_test_accuracy"] == 0.95

    def test_is_complete(self, mnist_runner):
        # Complete
        complete_result = {"epoch_test_accuracies": [0.9] * 5, "expected_epochs": 5}
        assert mnist_runner.is_complete(complete_result) is True
        # Incomplete
        incomplete_result = {"epoch_test_accuracies": [0.9] * 4, "expected_epochs": 5}
        assert mnist_runner.is_complete(incomplete_result) is False
        # More epochs than expected is still complete
        over_result = {"epoch_test_accuracies": [0.9] * 6, "expected_epochs": 5}
        assert mnist_runner.is_complete(over_result) is True
        # Missing key
        assert mnist_runner.is_complete({}) is False
        assert mnist_runner.is_complete({"epoch_test_accuracies": [0.9] * 5}) is True  # uses default


class TestSyntheticFixedTimeTrialRunnerUnit:
    """Unit tests for the SyntheticFixedTimeTrialRunner class."""

    @pytest.fixture
    def sft_runner(self):
        context = MagicMock()
        context.num_steps = 1000
        context.experiment.optimizer = OptimizerType.SGD
        runner = SyntheticFixedTimeTrialRunner(context)
        return runner

    def test_get_snapshot_steps(self, sft_runner):
        steps = sft_runner._get_snapshot_steps(max_steps=150)
        # Based on the 1,2,5 pattern
        expected = {0, 1, 2, 5, 10, 20, 50, 100, 149}
        assert set(steps) == expected

    def test_should_save_checkpoint(self, sft_runner):
        sft_runner.snapshot_steps = {0, 10, 100, 999}
        assert sft_runner._should_save_checkpoint(step=10) is True
        assert sft_runner._should_save_checkpoint(step=999) is True
        assert sft_runner._should_save_checkpoint(step=50) is False

    def test_capture_iterator_state(self, sft_runner):
        mock_iterator = Mock()
        mock_iterator.current_batch_key_seed = 12345
        results = {}
        updated_results = sft_runner._capture_iterator_state(mock_iterator, results)
        assert updated_results["batch_key_seed"] == 12345

    def test_is_complete(self, sft_runner):
        # Complete
        complete_result = {"loss_history": [0.1] * 1000, "expected_steps": 1000}
        assert sft_runner.is_complete(complete_result) is True
        # Incomplete
        incomplete_result = {"loss_history": [0.1] * 999, "expected_steps": 1000}
        assert sft_runner.is_complete(incomplete_result) is False
        # More steps than expected is still complete
        over_result = {"loss_history": [0.1] * 1001, "expected_steps": 1000}
        assert sft_runner.is_complete(over_result) is True
        # Missing key
        assert sft_runner.is_complete({}) is False
        assert sft_runner.is_complete({"loss_history": [0.1] * 1000}) is True  # uses default


class TestSyntheticFixedDataTrialRunnerUnit:
    """Unit tests for the SyntheticFixedDataTrialRunner class."""

    @pytest.fixture
    def sfd_runner(self):
        context = MagicMock()
        context.num_epochs = 5
        context.run_key = RunKey(batch_size=10, eta=0.1)
        context.train_ds = (np.zeros((100, 10)), np.zeros((100, 1)))  # P=100
        context.num_steps = 50  # 5 epochs * (100 samples / 10 batch_size) = 50 steps
        context.experiment.optimizer = OptimizerType.SGD
        runner = SyntheticFixedDataTrialRunner(context)
        return runner

    def test_init_calculates_steps(self, sfd_runner):
        assert sfd_runner.steps_per_epoch == 10  # 100 // 10
        # snapshot steps are calculated on total steps
        assert sfd_runner.snapshot_steps is not None

    def test_post_step_hook(self, sfd_runner):
        results = {"epoch": -1}
        # Not end of epoch
        updated_results = sfd_runner._post_step_hook(step=8, params="dummy", results=results)
        assert updated_results["epoch"] == -1  # Unchanged

        # End of first epoch (step 9 is 10th step)
        updated_results = sfd_runner._post_step_hook(step=9, params="dummy", results=results)
        assert updated_results["epoch"] == 0

        # End of second epoch (step 19 is 20th step)
        updated_results = sfd_runner._post_step_hook(step=19, params="dummy", results=results)
        assert updated_results["epoch"] == 1
