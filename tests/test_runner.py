from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.runner import (
    EtaStabilityTracker,
    ExperimentTypeChecker,
    RunStatus,
    _get_trial_runner,
    compute_model_widths,
    compute_num_steps,
    initialize_model_params,
    initialize_results_and_checkpoints,
    prepare_datasets,
    should_skip_batch_size,
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
    mock_exp = Mock()
    mock_exp.gamma = 1.0
    mock_exp.L = 2
    mock_exp.optimizer = OptimizerType.SGD
    mock_exp.parameterization = Parameterization.SP
    mock_exp.N = 128
    # Add attributes needed by specific runners to avoid TypeErrors
    mock_exp.loss_type = LossType.XENT
    mock_exp.num_outputs = 10
    mock_exp.P = 1000
    return mock_exp


@pytest.fixture
def base_runner_kwargs(mock_experiment, mock_run_key):
    """Provides a dictionary of base keyword arguments for creating a TrialRunner."""
    return {
        "experiment": mock_experiment,
        "run_key": mock_run_key,
        "params0": Mock(),
        "mlp_instance": Mock(),
        "checkpoint_manager": Mock(),
        "pbar": Mock(),
        "no_save": True,
        "init_key": 0,
    }


@pytest.fixture
def validation_setup():
    """Provides a standard setup for testing validate_and_store_result."""
    from batch_size_studies.experiments import SyntheticExperimentFixedData

    mock_exp = Mock(spec=SyntheticExperimentFixedData)
    mock_exp.num_epochs = 4  # A default for MNIST tests

    checkpoint_manager = Mock()
    checkpoint_manager.exp_dir = "/fake/path"

    return SimpleNamespace(
        mock_exp=mock_exp,
        type_checker=ExperimentTypeChecker(mock_exp),
        run_key=RunKey(batch_size=32, eta=0.1),
        result={"loss_history": [1.0, 0.9, 0.8]},
        results_dict={},
        failed_runs=set(),
        checkpoint_manager=checkpoint_manager,
    )


# ============================================================================
# TESTS FOR ExperimentTypeChecker
# ============================================================================


class TestExperimentTypeChecker:
    """Tests for experiment type detection."""

    def test_mnist_detection(self):
        """Test that MNIST experiments are correctly identified."""
        from batch_size_studies.experiments import MNISTExperiment

        real_mnist = Mock(spec=MNISTExperiment)
        checker = ExperimentTypeChecker(real_mnist)

        assert checker.is_mnist is True
        assert checker.is_synthetic_fixed_data is False
        assert checker.is_synthetic_fixed_time is False
        assert checker.uses_dataset is True

    def test_synthetic_fixed_data_detection(self):
        """Test that synthetic fixed-data experiments are correctly identified."""
        from batch_size_studies.experiments import SyntheticExperimentFixedData

        synthetic = Mock(spec=SyntheticExperimentFixedData)
        checker = ExperimentTypeChecker(synthetic)

        assert checker.is_mnist is False
        assert checker.is_synthetic_fixed_data is True
        assert checker.is_synthetic_fixed_time is False
        assert checker.uses_dataset is True

    def test_synthetic_fixed_time_detection(self):
        """Test that synthetic fixed-time experiments are correctly identified."""
        from batch_size_studies.experiments import SyntheticExperimentFixedTime

        synthetic = Mock(spec=SyntheticExperimentFixedTime)
        checker = ExperimentTypeChecker(synthetic)

        assert checker.is_mnist is False
        assert checker.is_synthetic_fixed_data is False
        assert checker.is_synthetic_fixed_time is True
        assert checker.uses_dataset is False


# ============================================================================
# TESTS FOR EtaStabilityTracker
# ============================================================================


class TestEtaStabilityTracker:
    """Tests for early stopping based on consecutive successes."""

    def test_threshold_reached_after_consecutive_successes(self):
        """Test that threshold is reached after exactly N consecutive successes."""
        tracker = EtaStabilityTracker(depth=3)

        assert tracker.update(is_successful=True) is False
        assert tracker.count == 1

        assert tracker.update(is_successful=True) is False
        assert tracker.count == 2

        assert tracker.update(is_successful=True) is True
        assert tracker.count == 3

    def test_failure_resets_counter(self):
        """Test that a failure resets the consecutive success counter."""
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
        """Test that tracker is disabled when depth is None."""
        tracker = EtaStabilityTracker(depth=None)

        # Should never trigger regardless of successes
        for _ in range(10):
            assert tracker.update(is_successful=True) is False

    def test_disabled_when_depth_is_zero(self):
        """Test that tracker is disabled when depth is zero."""
        tracker = EtaStabilityTracker(depth=0)

        assert tracker.update(is_successful=True) is False

    def test_reset_clears_counter(self):
        """Test that reset properly clears the counter."""
        tracker = EtaStabilityTracker(depth=3)

        tracker.update(is_successful=True)
        tracker.update(is_successful=True)
        assert tracker.count == 2

        tracker.reset()
        assert tracker.count == 0

    def test_handles_alternating_success_failure(self):
        """Test behavior with alternating success/failure pattern."""
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
        """Test that MNIST experiments use num_outputs for final layer."""
        from batch_size_studies.experiments import MNISTExperiment

        mock_mnist = Mock(spec=MNISTExperiment)
        mock_mnist.D = 784
        mock_mnist.N = 128
        mock_mnist.L = 3
        mock_mnist.num_outputs = 10

        type_checker = ExperimentTypeChecker(mock_mnist)
        widths = compute_model_widths(mock_mnist, type_checker)

        # [input, hidden1, hidden2, output]
        assert widths == [784, 128, 128, 10]

    def test_synthetic_widths_use_single_output(self):
        """Test that synthetic experiments use single output dimension."""
        from batch_size_studies.experiments import SyntheticExperimentFixedData

        mock_synthetic = Mock(spec=SyntheticExperimentFixedData)
        mock_synthetic.D = 50
        mock_synthetic.N = 32
        mock_synthetic.L = 2

        type_checker = ExperimentTypeChecker(mock_synthetic)
        widths = compute_model_widths(mock_synthetic, type_checker)

        # [input, hidden1, output=1]
        assert widths == [50, 32, 1]

    def test_deep_network_architecture(self):
        """Test width computation for deeper networks."""
        from batch_size_studies.experiments import MNISTExperiment

        mock_exp = Mock(spec=MNISTExperiment)
        mock_exp.D = 100
        mock_exp.N = 64
        mock_exp.L = 5  # Deeper network
        mock_exp.num_outputs = 10

        type_checker = ExperimentTypeChecker(mock_exp)
        widths = compute_model_widths(mock_exp, type_checker)

        # [input, hidden1, hidden2, hidden3, hidden4, output]
        assert len(widths) == 6
        assert widths == [100, 64, 64, 64, 64, 10]


# ============================================================================
# TESTS FOR compute_num_steps
# ============================================================================


class TestComputeNumSteps:
    """Tests for training step computation."""

    def test_fixed_time_experiment_uses_num_steps(self):
        """Test that fixed-time experiments use their num_steps directly."""
        from batch_size_studies.experiments import SyntheticExperimentFixedTime

        mock_exp = Mock(spec=SyntheticExperimentFixedTime)
        mock_exp.num_steps = 10000

        type_checker = ExperimentTypeChecker(mock_exp)

        # batch_size and train_ds shouldn't matter
        num_steps = compute_num_steps(mock_exp, type_checker, batch_size=64, train_ds=None)

        assert num_steps == 10000

    def test_mnist_computation_with_default_epochs(self):
        """Test step computation for MNIST with default epochs."""
        from batch_size_studies.experiments import MNISTExperiment

        mock_exp = Mock(spec=MNISTExperiment)
        mock_exp.num_epochs = 5

        type_checker = ExperimentTypeChecker(mock_exp)

        train_ds = {"image": np.zeros((1000, 784)), "label": np.zeros(1000)}
        batch_size = 32

        num_steps = compute_num_steps(mock_exp, type_checker, batch_size=batch_size, train_ds=train_ds)

        # 1000 samples / 32 batch_size = 31 steps per epoch
        # 31 * 5 epochs = 155 steps
        expected_steps = (1000 // 32) * 5
        assert num_steps == expected_steps

    def test_synthetic_computation_with_custom_epochs(self):
        """Test step computation with custom num_epochs in kwargs."""
        from batch_size_studies.experiments import SyntheticExperimentFixedData

        mock_exp = Mock(spec=SyntheticExperimentFixedData)
        mock_exp.num_epochs = 3  # Default in experiment
        mock_exp.P = 5000  # Dataset size

        type_checker = ExperimentTypeChecker(mock_exp)

        train_ds = (np.zeros((5000, 10)), np.zeros(5000))
        batch_size = 100

        num_steps = compute_num_steps(
            mock_exp,
            type_checker,
            batch_size=batch_size,
            train_ds=train_ds,
            num_epochs=10,  # Custom override
        )

        # 5000 / 100 = 50 steps per epoch
        # 50 * 10 epochs = 500 steps
        assert num_steps == 500

    def test_edge_case_single_batch_per_epoch(self):
        """Test computation when dataset size equals batch size."""
        from batch_size_studies.experiments import MNISTExperiment

        mock_exp = Mock(spec=MNISTExperiment)
        mock_exp.num_epochs = 3

        type_checker = ExperimentTypeChecker(mock_exp)

        train_ds = {"image": np.zeros((128, 784)), "label": np.zeros(128)}
        batch_size = 128

        num_steps = compute_num_steps(mock_exp, type_checker, batch_size=batch_size, train_ds=train_ds)

        # 1 step per epoch * 3 epochs = 3 steps
        assert num_steps == 3


# ============================================================================
# TESTS FOR should_skip_batch_size
# ============================================================================


class TestShouldSkipBatchSize:
    """Tests for batch size validation."""

    def test_skip_when_batch_size_exceeds_mnist_dataset(self):
        """Test that oversized batch sizes are skipped for MNIST."""
        from batch_size_studies.experiments import MNISTExperiment

        mock_exp = Mock(spec=MNISTExperiment)
        type_checker = ExperimentTypeChecker(mock_exp)

        train_ds = {"image": np.zeros((1000, 784)), "label": np.zeros(1000)}

        # Batch size larger than dataset
        should_skip = should_skip_batch_size(
            batch_size=2000, train_ds=train_ds, type_checker=type_checker, experiment=mock_exp
        )

        assert should_skip is True

    def test_allow_valid_batch_size_for_synthetic(self):
        """Test that valid batch sizes are allowed."""
        from batch_size_studies.experiments import SyntheticExperimentFixedData

        mock_exp = Mock(spec=SyntheticExperimentFixedData)
        mock_exp.P = 5000
        type_checker = ExperimentTypeChecker(mock_exp)

        train_ds = (np.zeros((5000, 10)), np.zeros(5000))

        should_skip = should_skip_batch_size(
            batch_size=128, train_ds=train_ds, type_checker=type_checker, experiment=mock_exp
        )

        assert should_skip is False

    def test_fixed_time_never_skips(self):
        """Test that fixed-time experiments never skip batch sizes."""
        from batch_size_studies.experiments import SyntheticExperimentFixedTime

        mock_exp = Mock(spec=SyntheticExperimentFixedTime)
        type_checker = ExperimentTypeChecker(mock_exp)

        # Even with ridiculous batch size, should not skip
        should_skip = should_skip_batch_size(
            batch_size=1_000_000, train_ds=None, type_checker=type_checker, experiment=mock_exp
        )

        assert should_skip is False

    def test_boundary_case_exact_match(self):
        """Test when batch size exactly equals dataset size."""
        from batch_size_studies.experiments import MNISTExperiment

        mock_exp = Mock(spec=MNISTExperiment)
        type_checker = ExperimentTypeChecker(mock_exp)

        train_ds = {"image": np.zeros((1000, 784)), "label": np.zeros(1000)}

        # Batch size exactly equals dataset size - should be allowed
        should_skip = should_skip_batch_size(
            batch_size=1000, train_ds=train_ds, type_checker=type_checker, experiment=mock_exp
        )

        assert should_skip is False


# ============================================================================
# TESTS FOR RunStatus
# ============================================================================


class TestRunStatus:
    """Tests for run status checking."""

    def test_should_run_when_no_save_enabled(self):
        """Test that runs always execute when no_save is True."""
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=True
        )

        assert status.should_run is True
        assert status.is_successful is False

    def test_skip_previously_failed_run(self):
        """Test that previously failed runs are skipped."""
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {}
        failed_runs = {run_key}

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is False
        assert status.is_successful is False

    def test_skip_completed_run(self):
        """Test that completed runs are skipped."""
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {run_key: {"loss_history": [1.0] * 1000}}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is False
        assert status.is_successful is True

    def test_run_incomplete_result(self):
        """Test that incomplete runs are re-executed."""
        run_key = RunKey(batch_size=32, eta=0.1)
        results_dict = {run_key: {"loss_history": [1.0] * 500}}
        failed_runs = set()

        status = RunStatus(
            run_key=run_key, results_dict=results_dict, failed_runs=failed_runs, num_steps=1000, no_save=False
        )

        assert status.should_run is True

    def test_run_missing_loss_history(self):
        """Test handling of results without loss_history key."""
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
        """Test that successful synthetic results are stored correctly."""
        s = validation_setup

        is_successful = validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            type_checker=s.type_checker,
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

    def test_failed_mnist_result_without_accuracy(self, validation_setup):
        """Test that MNIST results without accuracy are marked as failed."""
        from batch_size_studies.experiments import MNISTExperiment

        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment)
        s.type_checker = ExperimentTypeChecker(s.mock_exp)
        s.result = {"loss_history": [1.0, 0.9, 0.8]}  # Missing final_test

        is_successful = validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            type_checker=s.type_checker,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        assert is_successful is False
        assert s.run_key not in s.results_dict
        assert s.run_key in s.failed_runs

    def test_mnist_result_with_nan_accuracy(self, validation_setup):
        """Test that MNIST results with NaN accuracy are marked as failed."""
        from batch_size_studies.experiments import MNISTExperiment

        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment)
        s.type_checker = ExperimentTypeChecker(s.mock_exp)
        s.result = {"loss_history": [1.0, 0.9, 0.8]}  # Missing final_test_accuracy

        is_successful = validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            type_checker=s.type_checker,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        assert is_successful is False
        assert s.run_key in s.failed_runs

    def test_checkpoint_cleanup_called_for_completed_runs(self, validation_setup):
        """Test that checkpoints are cleaned up for completed runs."""
        s = validation_setup

        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            type_checker=s.type_checker,
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
        from batch_size_studies.experiments import MNISTExperiment

        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment)
        s.mock_exp.num_epochs = 4
        s.type_checker = ExperimentTypeChecker(s.mock_exp)
        s.result = {"final_test_accuracy": 0.9, "epoch_test_accuracies": [0.8, 0.85, 0.88, 0.9]}

        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            type_checker=s.type_checker,
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
        from batch_size_studies.experiments import MNISTExperiment

        s = validation_setup
        s.mock_exp = Mock(spec=MNISTExperiment)
        s.mock_exp.num_epochs = 4
        s.type_checker = ExperimentTypeChecker(s.mock_exp)
        s.result = {"final_test_accuracy": 0.9, "epoch_test_accuracies": [0.8, 0.85]}

        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            type_checker=s.type_checker,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=False,
        )

        # Should NOT be called because run is not fully complete
        s.checkpoint_manager.cleanup_live_checkpoint.assert_not_called()

    def test_previous_result_removed_on_failure(self, validation_setup):
        """Test that previous results are removed when run fails."""
        s = validation_setup
        s.results_dict[s.run_key] = {"old_result": "data"}

        validate_and_store_result(
            result=None,  # Failed run
            run_key=s.run_key,
            type_checker=s.type_checker,
            results_dict=s.results_dict,
            failed_runs=s.failed_runs,
            experiment=s.mock_exp,
            checkpoint_manager=s.checkpoint_manager,
            no_save=True,
        )

        assert s.run_key not in s.results_dict
        assert s.run_key in s.failed_runs

    def test_failed_run_discarded_from_failed_set_on_success(self, validation_setup):
        """Test that previously failed runs are removed from failed set on success."""
        s = validation_setup
        s.failed_runs = {s.run_key}  # Previously failed

        validate_and_store_result(
            result=s.result,
            run_key=s.run_key,
            type_checker=s.type_checker,
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
        """Test that in no_save mode, new empty results are created."""
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
        """Test that in load mode, existing results are loaded."""
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
        """Test that no_save=True always generates new parameters directly."""
        mock_mlp = Mock()
        mock_mlp.init_params.return_value = "new_params"
        mock_manager = Mock()

        params = initialize_model_params(mock_mlp, mock_manager, init_key=0, widths=[10, 1], no_save=True)

        assert params == "new_params"
        mock_mlp.init_params.assert_called_once_with(0, [10, 1])
        # Ensure the manager's more complex logic is not invoked
        mock_manager.initialize_and_save_initial_params.assert_not_called()

    def test_save_mode_delegates_to_manager(self):
        """Test that in save mode, parameter initialization is delegated to the CheckpointManager."""
        mock_mlp = Mock()
        mock_manager = Mock()
        mock_manager.initialize_and_save_initial_params.return_value = "params_from_manager"

        params = initialize_model_params(mock_mlp, mock_manager, init_key=42, widths=[10, 1], no_save=False)

        assert params == "params_from_manager"
        mock_manager.initialize_and_save_initial_params.assert_called_once_with(42, mock_mlp, [10, 1])
        mock_mlp.init_params.assert_not_called()


# ============================================================================
# TESTS FOR _get_trial_runner
# ============================================================================


class TestGetTrialRunner:
    """Tests for the trial runner factory function."""

    def test_returns_mnist_runner(self, base_runner_kwargs):
        """Test that it returns MNISTTrialRunner for MNIST experiments."""
        mock_type_checker = Mock()
        mock_type_checker.is_mnist = True
        mock_type_checker.is_synthetic_fixed_data = False
        mock_type_checker.is_synthetic_fixed_time = False

        runner_kwargs = base_runner_kwargs.copy()
        # Configure mocks to be dictionary-like and have a shape attribute
        mock_train_ds = {"image": Mock(), "label": Mock()}
        mock_train_ds["image"].shape = (100,)
        mock_test_ds = {"image": Mock(), "label": Mock()}
        mock_test_ds["image"].shape = (100,)
        runner_kwargs.update(
            {
                "train_ds": mock_train_ds,
                "test_ds": mock_test_ds,
            }
        )

        runner = _get_trial_runner(mock_type_checker, **runner_kwargs)
        assert isinstance(runner, MNISTTrialRunner)

    def test_returns_synthetic_fixed_data_runner(self, base_runner_kwargs):
        """Test that it returns SyntheticFixedDataTrialRunner for its experiment type."""
        mock_type_checker = Mock()
        mock_type_checker.is_mnist = False
        mock_type_checker.is_synthetic_fixed_data = True
        mock_type_checker.is_synthetic_fixed_time = False

        runner_kwargs = base_runner_kwargs.copy()
        mock_x_data = Mock()
        mock_x_data.shape = (100,)
        runner_kwargs.update(
            {
                "X_data": mock_x_data,
                "y_data": Mock(),
                "num_epochs": 1,
            }
        )
        runner = _get_trial_runner(mock_type_checker, **runner_kwargs)
        assert isinstance(runner, SyntheticFixedDataTrialRunner)

    def test_returns_synthetic_fixed_time_runner(self, base_runner_kwargs):
        """Test that it returns SyntheticFixedTimeTrialRunner for its experiment type."""
        mock_type_checker = Mock()
        mock_type_checker.is_mnist = False
        mock_type_checker.is_synthetic_fixed_data = False
        mock_type_checker.is_synthetic_fixed_time = True

        runner_kwargs = base_runner_kwargs.copy()
        runner_kwargs["num_steps"] = 100
        runner = _get_trial_runner(mock_type_checker, **runner_kwargs)
        assert isinstance(runner, SyntheticFixedTimeTrialRunner)

    def test_returns_none_for_unknown_type(self, caplog):
        """Test that it returns None and logs an error for unknown experiment types."""
        mock_type_checker = Mock()
        mock_type_checker.is_mnist = False
        mock_type_checker.is_synthetic_fixed_data = False
        mock_type_checker.is_synthetic_fixed_time = False

        mock_experiment = Mock()
        mock_experiment.experiment_type = "future_experiment"

        runner = _get_trial_runner(mock_type_checker, experiment=mock_experiment)
        assert runner is None
        assert "Unknown experiment type" in caplog.text


# ============================================================================
# TESTS FOR prepare_datasets
# ============================================================================


class TestPrepareDatasets:
    """Tests for the data loading helper function."""

    @patch("batch_size_studies.runner.load_datasets")
    def test_loads_standard_mnist(self, mock_loader):
        """Test loading for a standard MNISTExperiment."""
        from batch_size_studies.experiments import MNISTExperiment

        mock_exp = Mock(spec=MNISTExperiment)
        type_checker = ExperimentTypeChecker(mock_exp)

        mock_loader.return_value = (
            (np.zeros((100, 28, 28, 1)), np.zeros(100)),
            (np.zeros((20, 28, 28, 1)), np.zeros(20)),
        )

        train_ds, test_ds = prepare_datasets(mock_exp, type_checker, init_key=0)

        mock_loader.assert_called_once()
        assert train_ds is not None and test_ds is not None
        assert len(train_ds["image"]) == 100
        assert len(test_ds["image"]) == 20

    @patch("batch_size_studies.runner.load_mnist1m_dataset")
    def test_loads_and_samples_mnist1m(self, mock_loader):
        """Test loading and sampling for MNIST1MSampledExperiment."""
        from batch_size_studies.experiments import MNIST1MSampledExperiment

        mock_exp = Mock(spec=MNIST1MSampledExperiment)
        mock_exp.max_train_samples = 50  # Sample down to 50

        type_checker = ExperimentTypeChecker(mock_exp)

        # Use arange to easily check if shuffling occurred
        mock_loader.return_value = (
            (np.arange(100).reshape(100, 1, 1, 1), np.arange(100)),
            (np.zeros((20, 28, 28, 1)), np.zeros(20)),
        )

        train_ds, test_ds = prepare_datasets(mock_exp, type_checker, init_key=42)

        mock_loader.assert_called_once()
        assert train_ds is not None and test_ds is not None
        assert len(train_ds["image"]) == 50
        # Check that it's not just the first 50 samples (i.e., it was shuffled)
        assert not np.array_equal(train_ds["image"].flatten(), np.arange(50))

    def test_generates_synthetic_data(self):
        """Test data generation for SyntheticExperimentFixedData."""
        from batch_size_studies.experiments import SyntheticExperimentFixedData

        mock_exp = Mock(spec=SyntheticExperimentFixedData)
        mock_exp.generate_data.return_value = (np.zeros((100, 10)), np.zeros(100))
        mock_exp.seed = 42

        type_checker = ExperimentTypeChecker(mock_exp)

        train_ds, test_ds = prepare_datasets(mock_exp, type_checker, init_key=0)

        mock_exp.generate_data.assert_called_once()
        assert test_ds is None
        assert isinstance(train_ds, tuple) and len(train_ds) == 2

    def test_handles_dataset_not_found(self, caplog):
        """Test graceful failure when a dataset file is not found."""
        from batch_size_studies.experiments import MNISTExperiment

        mock_exp = Mock(spec=MNISTExperiment)
        type_checker = ExperimentTypeChecker(mock_exp)

        # Mock the loader to raise FileNotFoundError
        mock_loader = Mock(side_effect=FileNotFoundError("Dataset file missing"))

        train_ds, test_ds = prepare_datasets(mock_exp, type_checker, init_key=0, dataset_loader=mock_loader)

        assert train_ds is None and test_ds is None
        assert "Dataset not found" in caplog.text

    def test_no_data_loaded_for_fixed_time(self):
        """Test that no data is loaded for fixed-time synthetic experiments."""
        from batch_size_studies.experiments import SyntheticExperimentFixedTime

        mock_exp = Mock(spec=SyntheticExperimentFixedTime)
        type_checker = ExperimentTypeChecker(mock_exp)

        train_ds, test_ds = prepare_datasets(mock_exp, type_checker, init_key=0)

        assert train_ds is None and test_ds is None
