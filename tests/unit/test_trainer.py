from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batch_size_studies.definitions import LossType, OptimizerType, RunKey
from batch_size_studies.trainer import (
    MNISTTrialRunner,
    SyntheticFixedDataTrialRunner,
    SyntheticFixedTimeTrialRunner,
    TrialRunner,
)


class SimpleExperiment:
    def __init__(self, optimizer=OptimizerType.SGD, loss_type=LossType.MSE, num_outputs=1, d=1, p=8):
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.num_outputs = num_outputs
        self.D = d
        self.P = p

    def get_adjusted_eta(self, eta: float) -> float:
        return eta


class MinimalCheckpointManager:
    def __init__(self, params):
        self.params = params
        self.saved_checkpoints = []
        self.saved_snapshots = []

    def load_live_checkpoint(self, run_key):
        legacy_state = {"loss_history": [], "expected_steps": 0}
        return self.params, "legacy_state", legacy_state, 0

    def save_live_checkpoint(self, run_key, step, params, opt_state, results):
        self.saved_checkpoints.append(step)

    def save_analysis_snapshot(self, run_key, step, params, params0):
        self.saved_snapshots.append(step)


@pytest.fixture(autouse=True)
def clear_jit_cache():
    TrialRunner.clear_cache()


class TestMNISTTrialRunnerUnit:
    """Unit tests for the MNISTTrialRunner class."""

    @pytest.fixture
    def mnist_runner(self):
        # Create a mock context object with necessary attributes
        context = MagicMock()
        context.run_key = RunKey(batch_size=64, eta=0.1)
        context.num_epochs = 5
        context.num_steps = 100
        context.train_ds = {"image": np.zeros((1280, 784))}  # 1280 samples
        context.test_ds = {"image": np.zeros((100, 784)), "label": np.zeros(100)}
        context.experiment = MagicMock()
        context.experiment.D = 784
        context.experiment.optimizer = OptimizerType.SGD
        context.experiment.loss_type = LossType.XENT
        context.kwargs = {}

        # Mock methods that would be called
        runner = MNISTTrialRunner(context)
        runner.eval_step = Mock(return_value=0.95)  # Mock the JITted eval step
        runner.pbar = Mock()  # Mock the progress bar
        return runner

    def test_snapshot_steps_include_epochs(self, mnist_runner):
        expected_epoch_end = {19, 39, 59, 79, 99}
        assert set(mnist_runner.snapshot_steps) >= expected_epoch_end

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

    def test_on_epoch_end(self, mnist_runner):
        params = "dummy_params"
        results = {"epoch_test_accuracies": []}

        # EVAL_BATCH_SIZE is 512. test_ds has 100 samples. So one loop.
        updated_results = mnist_runner._on_epoch_end(epoch=0, params=params, results=results, aux=None)

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


class TestSyntheticFixedTimeEvalDataset:
    def test_eval_dataset_skipped_for_mock_key(self):
        context = MagicMock()
        context.num_steps = 10
        context.init_key = MagicMock()
        context.experiment.optimizer = OptimizerType.SGD
        context.kwargs = {}
        runner = SyntheticFixedTimeTrialRunner(context)
        assert runner.eval_ds is None


class TestSyntheticFixedTimeTrialRunnerUnit:
    """Unit tests for the SyntheticFixedTimeTrialRunner class."""

    @pytest.fixture
    def sft_runner(self):
        context = MagicMock()
        context.num_steps = 1000
        context.experiment.optimizer = OptimizerType.SGD
        context.kwargs = {}
        runner = SyntheticFixedTimeTrialRunner(context)
        return runner

    def test_get_snapshot_steps(self, sft_runner):
        steps = sft_runner._compute_snapshot_steps(max_steps=150, dense=True)
        # Based on the 1,2,5 pattern
        expected = {0, 1, 2, 5, 10, 20, 50, 100, 149}
        assert set(steps) == expected

    def test_get_snapshot_steps_sparse(self, sft_runner):
        steps = sft_runner._compute_snapshot_steps(max_steps=150, dense=False)
        assert steps == [0, 149]

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
        context.kwargs = {}
        runner = SyntheticFixedDataTrialRunner(context)
        return runner

    def test_init_calculates_steps(self, sfd_runner):
        assert sfd_runner.steps_per_epoch == 10  # 100 // 10
        # snapshot steps are calculated on total steps
        assert sfd_runner.snapshot_steps is not None
        expected_epochs = {9, 19, 29, 39, 49}
        assert set(sfd_runner.snapshot_steps) >= expected_epochs

    def test_post_step_hook(self, sfd_runner):
        results = {"epoch": -1}
        # Not end of epoch
        updated_results = sfd_runner._post_step_hook(step=8, params="dummy", results=results, aux=None)
        assert updated_results["epoch"] == -1  # Unchanged
        iterator_state = updated_results["iterator_state"]
        assert iterator_state["global_step"] == 9
        assert iterator_state["epoch_seed"] == sfd_runner._get_epoch_seed(iterator_state["epoch"])

        # End of first epoch (step 9 is 10th step)
        updated_results = sfd_runner._post_step_hook(step=9, params="dummy", results=results, aux=None)
        assert updated_results["epoch"] == 0
        assert updated_results["iterator_state"]["step_in_epoch"] == 0
        assert "step_in_epoch" in updated_results["iterator_state"]

        # End of second epoch (step 19 is 20th step)
        updated_results = sfd_runner._post_step_hook(step=19, params="dummy", results=results, aux=None)
        assert updated_results["epoch"] == 1

    def test_adjust_start_step_reads_iterator_state(self, sfd_runner):
        results = {"iterator_state": {"global_step": 7}}
        adjusted = sfd_runner._adjust_start_step(start_step=0, results=results)
        assert adjusted == 7

    def test_post_training_hook_removes_iterator_state(self, sfd_runner):
        results = {"iterator_state": {"global_step": 5}, "epoch_test_accuracies": []}
        cleaned = sfd_runner._post_training_hook(params="dummy", results=results)
        assert "iterator_state" not in cleaned


class MinimalTrialRunner(TrialRunner):
    def _init_results(self) -> dict:
        return {"loss_history": [], "expected_steps": self.num_steps}

    def _create_loss_fn(self):
        def loss_fn(params, x_batch, y_batch):
            return jnp.array(0.0), None

        return loss_fn

    def _create_jitted_update_step(self, loss_fn, base_optimizer_transform):
        @jax.jit
        def update_step_fn(params, opt_state, x_batch, y_batch, lr):
            return params, opt_state, jnp.array(0.0), None

        return update_step_fn

    def _create_data_iterator(self, start_step: int, results: dict):
        return iter([])

    def is_complete(self, result: dict) -> bool:
        return True


class SmallEvalSyntheticRunner(SyntheticFixedTimeTrialRunner):
    EVAL_MAX_SAMPLES = 5


def make_mnist_context(experiment, model_instance, params0):
    train_images = np.zeros((64, experiment.D))
    test_images = np.zeros((32, experiment.D))
    train_ds = {"image": train_images, "label": np.zeros(64, dtype=int)}
    test_ds = {"image": test_images, "label": np.zeros(32, dtype=int)}
    return SimpleNamespace(
        experiment=experiment,
        run_key=RunKey(batch_size=64, eta=0.1),
        params0=params0,
        model_instance=model_instance,
        no_save=True,
        checkpoint_manager=None,
        pbar=None,
        kwargs={},
        num_steps=0,
        num_epochs=1,
        train_ds=train_ds,
        test_ds=test_ds,
        init_key=0,
    )


def make_sft_context(experiment, init_key=0):
    params0 = jnp.zeros((1,))

    def model_instance(params, inputs):
        return jnp.zeros((inputs.shape[0], 1))

    return SimpleNamespace(
        experiment=experiment,
        run_key=RunKey(batch_size=1, eta=0.1),
        params0=params0,
        model_instance=model_instance,
        no_save=True,
        checkpoint_manager=None,
        pbar=None,
        kwargs={},
        num_steps=1,
        num_epochs=1,
        train_ds=None,
        test_ds=None,
        init_key=init_key,
    )


def test_trialrunner_migrates_legacy_opt_state(caplog):
    experiment = SimpleExperiment()
    params0 = jnp.zeros((1,))
    checkpoint_manager = MinimalCheckpointManager(jnp.ones((1,)))
    context = SimpleNamespace(
        experiment=experiment,
        run_key=RunKey(batch_size=1, eta=0.1),
        params0=params0,
        model_instance=lambda params, inputs: jnp.zeros((inputs.shape[0], 1)),
        no_save=False,
        checkpoint_manager=checkpoint_manager,
        pbar=None,
        kwargs={},
        num_steps=0,
        num_epochs=1,
        train_ds=None,
        test_ds=None,
        init_key=0,
    )

    with caplog.at_level("INFO"):
        runner = MinimalTrialRunner(context)
        result = runner.run()

    assert result == {"loss_history": [], "expected_steps": 0}
    assert any("Migrating old optimizer state format" in msg for msg in caplog.messages)
    assert runner._should_save_checkpoint(0) is False


def test_mnist_trialrunner_reuses_cached_eval_step(monkeypatch):
    TrialRunner.clear_cache()
    experiment = SimpleExperiment(optimizer=OptimizerType.SGD, loss_type=LossType.XENT, num_outputs=10, d=784)
    model_instance = lambda params, x: jnp.zeros((x.shape[0], experiment.num_outputs))
    params0 = {"w": jnp.zeros((experiment.D, experiment.num_outputs)), "b": jnp.zeros((experiment.num_outputs,))}

    context1 = make_mnist_context(experiment, model_instance, params0)
    runner1 = MNISTTrialRunner(context1)
    assert hasattr(runner1, "eval_step")

    def fail_create_eval(self):
        raise AssertionError("Should not be called when cache is hit")

    monkeypatch.setattr(MNISTTrialRunner, "_create_eval_step", fail_create_eval)
    context2 = make_mnist_context(experiment, model_instance, params0)
    runner2 = MNISTTrialRunner(context2)
    assert hasattr(runner2, "eval_step")


def test_synthetic_eval_dataset_without_generate_data():
    experiment = SimpleExperiment()
    context = make_sft_context(experiment, init_key=0)
    runner = SyntheticFixedTimeTrialRunner(context)
    assert runner.eval_ds is None


def test_synthetic_eval_dataset_handles_type_error():
    class FailingExperiment(SimpleExperiment):
        def generate_data(self, key):
            raise TypeError("missing argument")

    experiment = FailingExperiment()
    context = make_sft_context(experiment, init_key=0)
    runner = SyntheticFixedTimeTrialRunner(context)
    assert runner.eval_ds is None


def test_synthetic_eval_dataset_respects_max_samples():
    class LargeExperiment(SimpleExperiment):
        def generate_data(self, key):
            X = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)
            y = X[:, :1]
            return X, y

    experiment = LargeExperiment()
    context = make_sft_context(experiment, init_key=3)
    runner = SmallEvalSyntheticRunner(context)
    assert runner.eval_ds is not None
    X_eval, y_eval = runner.eval_ds
    assert X_eval.shape[0] == SmallEvalSyntheticRunner.EVAL_MAX_SAMPLES
    assert y_eval.shape[0] == SmallEvalSyntheticRunner.EVAL_MAX_SAMPLES
