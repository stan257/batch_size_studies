from types import SimpleNamespace
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import (
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
    SyntheticExperimentMLPTeacher,
)
from batch_size_studies.models import MLP
from batch_size_studies.runner import (
    CenteredModel,
    EtaStabilityTracker,
    RunStatus,
    _get_trial_runner,
    _run_single_trial,
    compute_model_widths,
    compute_num_steps,
    initialize_model_params,
    initialize_results_and_checkpoints,
    validate_and_store_result,
)
from batch_size_studies.trainer import (
    MNISTTrialRunner,
    SyntheticFixedDataTrialRunner,
    SyntheticFixedTimeTrialRunner,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_run_key():
    """Provides a mock RunKey with default values."""
    mock_key = Mock(spec=RunKey)
    mock_key.batch_size = 32
    mock_key.eta = 0.01
    return mock_key


@pytest.fixture
def mock_experiment():
    """Provides a generic mock experiment with common attributes."""
    # Use a real experiment type to satisfy isinstance checks
    mock_exp = Mock(spec=MNISTExperiment)
    mock_exp.gamma = 1.0
    mock_exp.L = 2
    mock_exp.optimizer = OptimizerType.SGD
    mock_exp.parameterization = Parameterization.SP
    mock_exp.N = 128
    mock_exp.loss_type = LossType.XENT
    mock_exp.num_outputs = 10
    mock_exp.P = 1000
    return mock_exp


@pytest.fixture
def base_runner_kwargs(mock_experiment, mock_run_key):
    """Provides a dictionary of base keyword arguments for creating a TrialRunner."""
    return {
        "run_key": mock_run_key,
        "params0": Mock(),
        "model_instance": Mock(),
        "checkpoint_manager": Mock(),
        "pbar": Mock(),
        "no_save": True,
        "init_key": 0,
    }


@pytest.fixture
def validation_setup():
    """Provides a standard setup for testing validate_and_store_result."""
    mock_exp = Mock(spec=SyntheticExperimentFixedData, num_epochs=4)

    checkpoint_manager = Mock()
    checkpoint_manager.exp_dir = "/fake/path"

    return SimpleNamespace(
        mock_exp=mock_exp,
        run_key=RunKey(batch_size=32, eta=0.1),
        result={"loss_history": [1.0, 0.9, 0.8]},
        results_dict={},
        failed_runs=set(),
        checkpoint_manager=checkpoint_manager,
    )


# ============================================================================
# TESTS FOR CenteredModel
# ============================================================================


class TestCenteredModel:
    """Tests for the CenteredModel wrapper class."""

    def test_output_is_zero_at_initialization(self):
        model_seed = 0
        data_key = jax.random.PRNGKey(1)

        mlp = MLP(parameterization=Parameterization.SP, gamma=1.0)
        widths = [10, 20, 5]
        params0 = mlp.init_params(model_seed, widths)

        centered_model = CenteredModel(model=mlp, params0=params0)

        dummy_input = jax.random.normal(data_key, (1, 10))
        uncentered_output = mlp(params0, dummy_input)
        assert uncentered_output.shape == (1, 5)
        assert not jnp.all(uncentered_output == 0), "Uncentered model output should not be zero at initialization"

        centered_model = CenteredModel(model=mlp, params0=params0)
        centered_output = centered_model(params0, dummy_input)
        # The output should be a zero vector of the correct shape
        assert centered_output.shape == (1, 5)
        assert jnp.all(centered_output == 0)


# ============================================================================
# TESTS FOR EtaStabilityTracker
# ============================================================================


class TestEtaStabilityTracker:
    """Tests for early stopping based on consecutive successes."""

    def test_threshold_reached_after_consecutive_successes(self):
        tracker = EtaStabilityTracker(depth=3)

        assert tracker.update(is_successful=True) is False
        assert tracker.count == 1

        assert tracker.update(is_successful=True) is False
        assert tracker.count == 2

        assert tracker.update(is_successful=True) is True
        assert tracker.count == 3

    def test_failure_resets_counter(self):
        tracker = EtaStabilityTracker(depth=3)

        tracker.update(is_successful=True)
        tracker.update(is_successful=True)
        assert tracker.count == 2

        tracker.update(is_successful=False)
        assert tracker.count == 0

        tracker.update(is_successful=True)
        tracker.update(is_successful=True)
        assert tracker.update(is_successful=True) is True

    def test_disabled_when_depth_is_none(self):
        tracker = EtaStabilityTracker(depth=None)

        # Should never trigger regardless of successes
        for _ in range(10):
            assert tracker.update(is_successful=True) is False

    def test_disabled_when_depth_is_zero(self):
        tracker = EtaStabilityTracker(depth=0)

        assert tracker.update(is_successful=True) is False

    def test_reset_clears_counter(self):
        tracker = EtaStabilityTracker(depth=3)

        tracker.update(is_successful=True)
        tracker.update(is_successful=True)
        assert tracker.count == 2

        tracker.reset()
        assert tracker.count == 0

    def test_handles_alternating_success_failure(self):
        tracker = EtaStabilityTracker(depth=3)

        # Alternating pattern should never reach threshold
        for _ in range(10):
            assert tracker.update(is_successful=True) is False
            tracker.update(is_successful=False)

        assert tracker.count == 0


# ============================================================================
# TESTS FOR compute_model_widths
# ============================================================================


class TestComputeModelWidths:
    """Tests for model architecture width computation."""

    def test_mnist_widths_include_output_dimension(self):
        mock_mnist = Mock(spec=MNISTExperiment)
        mock_mnist.D = 784
        mock_mnist.N = 128
        mock_mnist.L = 3
        mock_mnist.num_outputs = 10

        widths = compute_model_widths(mock_mnist)

        # [input, hidden1, hidden2, output]
        assert widths == [784, 128, 128, 10]

    def test_synthetic_widths_use_single_output(self):
        mock_synthetic = Mock(spec=SyntheticExperimentFixedData)
        mock_synthetic.D = 50
        mock_synthetic.N = 32
        mock_synthetic.L = 2

        widths = compute_model_widths(mock_synthetic)

        # [input, hidden1, output=1]
        assert widths == [50, 32, 1]

    def test_deep_network_architecture(self):
        mock_exp = Mock(spec=MNISTExperiment)
        mock_exp.D = 100
        mock_exp.N = 64
        mock_exp.L = 5  # Deeper network
        mock_exp.num_outputs = 10

        widths = compute_model_widths(mock_exp)

        # [input, hidden1, hidden2, hidden3, hidden4, output]
        assert len(widths) == 6
        assert widths == [100, 64, 64, 64, 64, 10]


# ============================================================================
# TESTS FOR compute_num_steps
# ============================================================================


class TestComputeNumSteps:
    """Tests for training step computation."""

    def test_fixed_time_experiment_uses_num_steps(self):
        mock_exp = Mock(spec=SyntheticExperimentFixedTime)
        mock_exp.num_steps = 10000

        # batch_size and train_ds shouldn't matter
        num_steps = compute_num_steps(mock_exp, batch_size=64, train_ds=None)

        assert num_steps == 10000

    def test_mnist_computation_with_default_epochs(self):
        mock_exp = Mock(spec=MNISTExperiment)
        mock_exp.num_epochs = 5

        train_ds = {"image": np.zeros((1000, 784)), "label": np.zeros(1000)}
        batch_size = 32

        num_steps = compute_num_steps(mock_exp, batch_size=batch_size, train_ds=train_ds)

        # 1000 samples / 32 batch_size = 31 steps per epoch
        # 31 * 5 epochs = 155 steps
        expected_steps = (1000 // 32) * 5
        assert num_steps == expected_steps

    def test_synthetic_computation_with_custom_epochs(self):
        mock_exp = Mock(spec=SyntheticExperimentFixedData)
        mock_exp.num_epochs = 3  # Default in experiment
        mock_exp.P = 5000  # Dataset size

        train_ds = (np.zeros((5000, 10)), np.zeros(5000))
        batch_size = 100

        num_steps = compute_num_steps(
            mock_exp,
            batch_size=batch_size,
            train_ds=train_ds,
            num_epochs=10,  # Custom override
        )

        # 5000 / 100 = 50 steps per epoch
        # 50 * 10 epochs = 500 steps
        assert num_steps == 500

    def test_edge_case_single_batch_per_epoch(self):
        mock_exp = Mock(spec=MNISTExperiment)
        mock_exp.num_epochs = 3

        train_ds = {"image": np.zeros((128, 784)), "label": np.zeros(128)}
        batch_size = 128

        num_steps = compute_num_steps(mock_exp, batch_size=batch_size, train_ds=train_ds)

        # 1 step per epoch * 3 epochs = 3 steps
        assert num_steps == 3


# ============================================================================
# TESTS FOR RunStatus
# ============================================================================


class TestRunStatus:
    """Tests for run status checking."""

    def test_should_run_when_no_save_enabled(self):
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=True
        )

        assert status.should_run is True
        assert status.is_successful is False

    def test_skip_previously_failed_run(self):
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {}
        failed_runs = {run_key}

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is False
        assert status.is_successful is False

    def test_skip_completed_run(self):
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {run_key: {"loss_history": [1.0] * 1000}}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is False
        assert status.is_successful is True

    def test_run_incomplete_result(self):
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {run_key: {"loss_history": [1.0] * 500}}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is True

    def test_run_missing_loss_history(self):
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {run_key: {"other_metric": 99}}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is True


# ============================================================================
# TESTS FOR validate_and_store_result
# ============================================================================


class TestValidateAndStoreResult:
    """Tests for result validation and storage."""

    def test_successful_synthetic_result_stored(self, validation_setup):
        s = validation_setup
        s.mock_exp.is_run_complete.return_value = True  # Simulate a complete run

        is_successful = validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        assert is_successful is True
        assert s.run_key in s.results_dict
        assert s.results_dict[s.run_key] == s.result
        assert s.run_key not in s.failed_runs

    def test_incomplete_run_is_stored_for_resumption(self, validation_setup):
        """
        Tests that an incomplete but valid run (e.g., finished early) is stored
        in the results dictionary but NOT marked as a failure. This is the
        correct behavior to allow for checkpoint resumption.
        """
        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment)
        s.mock_exp.is_run_complete.return_value = False  # Mark as incomplete
        s.result = {"loss_history": [1.0, 0.9, 0.8]}  # A valid, partial result

        is_successful = validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        # The run is not "successful" because it's not complete
        assert is_successful is False
        # CRUCIAL: The result should be stored to allow for resumption...
        assert s.run_key in s.results_dict
        # ...but it should NOT be added to the set of hard failures.
        assert s.run_key not in s.failed_runs

    def test_mnist_result_with_nan_accuracy(self, validation_setup):
        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment)
        s.mock_exp.is_run_complete.return_value = True  # Structurally complete
        s.result = {"final_test_accuracy": np.nan}

        is_successful = validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        assert is_successful is False
        assert s.run_key in s.failed_runs

    def test_checkpoint_cleanup_called_for_completed_runs(self, validation_setup):
        s = validation_setup

        s.mock_exp.is_run_complete.return_value = True
        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=False,  # Enable saving
        )

        # Should cleanup checkpoint for successful synthetic run
        s.checkpoint_manager.cleanup_live_checkpoint.assert_called_once_with(s.run_key)

    def test_checkpoint_cleanup_called_for_full_mnist_run(self, validation_setup):
        """Test that checkpoints ARE cleaned up for fully completed MNIST runs."""
        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment, num_epochs=4)
        s.mock_exp.is_run_complete.return_value = True
        s.result = {"final_test_accuracy": 0.9, "epoch_test_accuracies": [0.8, 0.85, 0.88, 0.9]}

        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=False,
        )

        # Should BE called because run is fully complete
        s.checkpoint_manager.cleanup_live_checkpoint.assert_called_once_with(s.run_key)

    def test_checkpoint_cleanup_not_called_for_partial_mnist_run(self, validation_setup):
        """Test that checkpoints are NOT cleaned up for partially completed MNIST runs."""
        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment, num_epochs=4)
        s.mock_exp.is_run_complete.return_value = False  # This run is incomplete
        s.result = {"final_test_accuracy": 0.9, "epoch_test_accuracies": [0.8, 0.85]}

        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=False,
        )

        # Should NOT be called because run is not fully complete
        s.checkpoint_manager.cleanup_live_checkpoint.assert_not_called()

    def test_previous_result_removed_on_failure(self, validation_setup):
        s = validation_setup
        s.mock_exp.is_run_complete.return_value = False
        s.results_dict[s.run_key] = {"old_result": "data"}

        validate_and_store_result(
            result=None,  # Failed run
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        assert s.run_key not in s.results_dict
        assert s.run_key in s.failed_runs

    def test_failed_run_discarded_from_failed_set_on_success(self, validation_setup):
        s = validation_setup
        s.failed_runs = {s.run_key}  # Previously failed
        s.mock_exp.is_run_complete.return_value = True

        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        assert s.run_key not in s.failed_runs
        assert s.run_key in s.results_dict


# ============================================================================
# TESTS FOR INITIALIZATION HELPERS
# ============================================================================


class TestInitializeResultsAndCheckpoints:
    """Tests for the results and checkpoint initialization helper."""

    def test_no_save_mode(self, tmp_path):
        mock_experiment = Mock()
        mock_experiment.load_results.return_value = ({"some": "data"}, {"a", "b"})
        mock_experiment.experiment_type = "test_exp"
        mock_experiment.to_params_dict.return_value = {}

        results, failed, manager = initialize_results_and_checkpoints(mock_experiment, str(tmp_path), no_save=True)

        assert results == {}
        assert failed == set()
        mock_experiment.load_results.assert_not_called()
        assert isinstance(manager, CheckpointManager)

    def test_load_mode(self, tmp_path):
        mock_experiment = Mock()
        mock_experiment.load_results.return_value = ({"some": "data"}, {"a", "b"})
        mock_experiment.experiment_type = "test_exp"
        mock_experiment.to_params_dict.return_value = {}

        results, failed, manager = initialize_results_and_checkpoints(mock_experiment, str(tmp_path), no_save=False)

        mock_experiment.load_results.assert_called_once_with(directory=str(tmp_path), silent=True)
        assert results == {"some": "data"}
        assert failed == {"a", "b"}
        assert isinstance(manager, CheckpointManager)


class TestInitializeModelParams:
    """Tests for the model parameter initialization helper."""

    def test_no_save_mode_always_initializes_directly(self):
        mock_mlp = Mock()
        mock_mlp.init_params.return_value = "new_params"
        mock_manager = Mock()

        params = initialize_model_params(mock_mlp, mock_manager, init_key=0, widths=[10, 1], no_save=True)

        assert params == "new_params"
        mock_mlp.init_params.assert_called_once_with(0, [10, 1])
        # Ensure the manager's more complex logic is not invoked
        mock_manager.initialize_and_save_initial_params.assert_not_called()

    def test_save_mode_delegates_to_manager(self):
        mock_mlp = Mock()
        mock_manager = Mock()
        mock_manager.initialize_and_save_initial_params.return_value = "params_from_manager"

        params = initialize_model_params(mock_mlp, mock_manager, init_key=42, widths=[10, 1], no_save=False)

        assert params == "params_from_manager"
        mock_manager.initialize_and_save_initial_params.assert_called_once_with(42, mock_mlp, [10, 1])
        mock_mlp.init_params.assert_not_called()


# ============================================================================
# TESTS FOR TRIAL EXECUTION HELPERS
# ============================================================================


class TestTrialExecutionHelpers:
    """Tests for the refactored trial execution helper functions."""

    @pytest.fixture
    def trial_setup(self, mock_experiment, mock_run_key):
        """Provides a common setup for trial execution tests."""
        mock_exp = Mock(spec=MNISTExperiment)
        mock_exp.num_epochs = 5
        mock_exp.parameterization = Parameterization.SP
        mock_exp.gamma = 1.0
        mock_exp.D = 784
        mock_exp.N = 128
        mock_exp.L = 2
        mock_exp.num_outputs = 10

        return SimpleNamespace(
            experiment=mock_exp,
            run_key=mock_run_key,
            results_dict={},
            failed_runs=set(),
            checkpoint_manager=Mock(spec=CheckpointManager),
            params0=Mock(),
            model_instance=Mock(),
            train_ds={"image": np.zeros((100, 784)), "label": np.zeros(100)},
            test_ds={"image": np.zeros((20, 784)), "label": np.zeros(20)},
            pbar=Mock(),
            no_save=True,
            init_key=0,
        )

    @patch("batch_size_studies.runner.RunStatus")
    def test_run_single_trial_skips_if_should_not_run(self, mock_run_status, trial_setup):
        mock_run_status.return_value.should_run = False
        mock_run_status.return_value.is_successful = True

        is_successful = _run_single_trial(**vars(trial_setup))

        assert is_successful is True
        mock_run_status.assert_called_once()

    @patch("batch_size_studies.runner.validate_and_store_result")
    @patch("batch_size_studies.runner._get_trial_runner")
    @patch("batch_size_studies.runner.RunStatus")
    def test_run_single_trial_executes_and_validates(
        self, mock_run_status, mock_get_runner, mock_validate, trial_setup
    ):
        mock_run_status.return_value.should_run = True
        mock_trial_runner = Mock()
        mock_trial_runner.run.return_value = {"loss": 0.1}
        mock_get_runner.return_value = mock_trial_runner
        mock_validate.return_value = True

        is_successful = _run_single_trial(**vars(trial_setup))

        assert is_successful is True
        mock_get_runner.assert_called_once()
        mock_trial_runner.run.assert_called_once()
        mock_validate.assert_called_once_with(
            {"loss": 0.1},
            trial_setup.run_key,
            trial_setup.results_dict,
            trial_setup.failed_runs,
            trial_setup.experiment,
            trial_setup.checkpoint_manager,
            trial_setup.no_save,
        )

    @patch("batch_size_studies.runner._get_trial_runner")
    @patch("batch_size_studies.runner.RunStatus")
    def test_run_single_trial_handles_runner_failure(self, mock_run_status, mock_get_runner, trial_setup):
        mock_run_status.return_value.should_run = True
        mock_get_runner.return_value = None  # Runner creation fails

        is_successful = _run_single_trial(**vars(trial_setup))

        assert is_successful is False
        mock_get_runner.assert_called_once()
        assert trial_setup.run_key in trial_setup.failed_runs


# ============================================================================
# TESTS FOR _get_trial_runner
# ============================================================================


class TestGetTrialRunner:
    """Tests for the trial runner factory function."""

    @pytest.mark.parametrize(
        "exp_spec, runner_class, extra_kwargs, loss_type",
        [
            (
                MNISTExperiment,
                MNISTTrialRunner,
                {
                    "train_ds": {"image": Mock(shape=(100,)), "label": Mock()},
                    "test_ds": {"image": Mock(shape=(100,)), "label": Mock()},
                },
                LossType.XENT,
            ),
            (
                SyntheticExperimentFixedData,
                SyntheticFixedDataTrialRunner,
                {"X_data": Mock(shape=(100,)), "y_data": Mock(), "num_epochs": 1},
                LossType.MSE,
            ),
            (
                SyntheticExperimentMLPTeacher,
                SyntheticFixedTimeTrialRunner,
                {"num_steps": 100},
                LossType.MSE,
            ),
        ],
    )
    def test_returns_correct_runner_for_experiment_type(
        self, base_runner_kwargs, exp_spec, runner_class, extra_kwargs, loss_type
    ):
        """Tests that _get_trial_runner returns the correct runner type for different experiments."""
        mock_experiment = Mock(spec=exp_spec)
        # Add required attributes for different experiment types
        mock_experiment.gamma = 1.0
        mock_experiment.L = 2
        mock_experiment.N = 32
        mock_experiment.parameterization = Parameterization.SP
        mock_experiment.optimizer = OptimizerType.SGD
        mock_experiment.loss_type = loss_type
        # Add seed attribute for synthetic experiments to avoid mock recursion
        # in the trial runner's __init__ method.
        if exp_spec == SyntheticExperimentFixedData:
            mock_experiment.seed = 0

        runner_kwargs = {**base_runner_kwargs, **extra_kwargs}

        mock_experiment.get_trial_runner_class.return_value = runner_class
        runner = _get_trial_runner(mock_experiment, **runner_kwargs)
        assert isinstance(runner, runner_class)

    def test_returns_none_for_unknown_type(self, caplog):
        mock_experiment = Mock()
        mock_experiment.experiment_type = "future_experiment"
