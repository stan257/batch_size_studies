from dataclasses import replace
from unittest.mock import Mock, patch

import jax.random as jr
import numpy as np
import pytest

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import (
    ExperimentBase,
    MNIST1MExperiment,
    MNIST1MSampledExperiment,
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
    SyntheticExperimentLinearTeacher,
    SyntheticExperimentMLPTeacher,
)
from batch_size_studies.models import MLP, LinearModel
from batch_size_studies.trainer import (
    MNISTTrialRunner,
    SyntheticFixedDataTrialRunner,
    SyntheticFixedTimeTrialRunner,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def fixed_time_config():
    """Fixture for a standard FixedTime experiment configuration."""
    return SyntheticExperimentFixedTime(
        D=16,
        P=128,
        N=32,
        K=2,
        num_steps=100,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )


@pytest.fixture
def fixed_data_config():
    """Fixture for a standard FixedData experiment configuration."""
    return SyntheticExperimentFixedData(
        D=16,
        P=128,
        N=32,
        K=2,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )


@pytest.fixture
def mlp_teacher_config():
    """A pytest fixture for a standard MLP Teacher experiment configuration."""
    return SyntheticExperimentMLPTeacher(
        D=16,
        P=128,
        N=32,
        L=2,
        gamma=1.0,
        parameterization=Parameterization.SP,
        num_steps=100,
        teacher_N=64,
        teacher_L=3,
        teacher_gamma=1.0,
        teacher_parameterization=Parameterization.SP,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )


@pytest.fixture
def linear_teacher_config():
    """Fixture for a standard Linear Teacher experiment configuration."""
    return SyntheticExperimentLinearTeacher(
        D=100,
        P=1000,
        alpha=1.0,
        beta=1.0,
        num_epochs=5,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )


@pytest.fixture
def mnist_config():
    """Fixture for a standard MNIST experiment configuration."""
    return MNISTExperiment(
        N=128,
        L=2,
        parameterization=Parameterization.SP,
        gamma=1.0,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.XENT,
        num_epochs=4,
    )


@pytest.fixture
def mnist1m_config():
    """Fixture for a standard MNIST-1M experiment configuration."""
    return MNIST1MExperiment(
        N=128,
        L=3,
        parameterization=Parameterization.MUP,
        gamma=1.0,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
        num_epochs=2,
    )


@pytest.fixture
def mnist1m_sampled_config():
    """Fixture for a standard sampled MNIST-1M experiment configuration."""
    return MNIST1MSampledExperiment(
        N=64,
        L=3,
        parameterization=Parameterization.MUP,
        gamma=2.0,
        optimizer=OptimizerType.ADAM,
        loss_type=LossType.XENT,
        num_epochs=5,
        max_train_samples=10000,
    )


# ============================================================================
# TEST CLASSES
# ============================================================================


class TestModelHandling:
    """Tests the polymorphic model handling methods on experiment classes."""

    def test_mlp_experiment_returns_correct_widths(self, mnist_config):
        """Verifies get_model_widths for an MLP experiment."""
        # mnist_config: D=784, N=128, L=2, num_outputs=10
        widths = mnist_config.get_model_widths()
        assert widths == [784, 128, 10]

    def test_linear_experiment_returns_correct_widths(self, linear_teacher_config):
        """Verifies get_model_widths for a Linear experiment."""
        # linear_teacher_config: D=100
        widths = linear_teacher_config.get_model_widths()
        assert widths == [100, 1]

    def test_mlp_experiment_returns_centered_model(self, mnist_config):
        """Verifies get_model_wrapper for an MLP experiment."""
        from batch_size_studies.runner import CenteredModel

        mock_model, mock_params0 = Mock(), Mock()
        wrapped_model = mnist_config.get_model_wrapper(mock_model, mock_params0)

        assert isinstance(wrapped_model, CenteredModel)
        assert wrapped_model.model is mock_model
        assert wrapped_model.params0 is mock_params0

    def test_linear_experiment_returns_raw_model(self, linear_teacher_config):
        """Verifies get_model_wrapper for a Linear experiment."""
        mock_model, mock_params0 = Mock(), Mock()
        wrapped_model = linear_teacher_config.get_model_wrapper(mock_model, mock_params0)
        assert wrapped_model is mock_model


class TestCreateModelInstance:
    """Tests the create_model_instance method on experiment classes."""

    def test_mlp_experiment_creates_mlp_model(self, mnist_config):
        """Verifies that an MLP-based experiment creates an MLP instance."""
        model = mnist_config.create_model_instance()
        assert isinstance(model, MLP)
        assert model.parameterization == mnist_config.parameterization
        assert model.gamma == mnist_config.gamma

    def test_linear_experiment_creates_linear_model(self, linear_teacher_config):
        """Verifies that a Linear-based experiment creates a LinearModel instance."""
        model = linear_teacher_config.create_model_instance()
        assert isinstance(model, LinearModel)


class TestGetTrialRunnerClass:
    """Tests the get_trial_runner_class method on experiment classes."""

    def test_mnist_returns_mnist_runner(self, mnist_config):
        assert mnist_config.get_trial_runner_class() is MNISTTrialRunner

    def test_mnist1m_returns_mnist_runner(self, mnist1m_config):
        assert mnist1m_config.get_trial_runner_class() is MNISTTrialRunner

    def test_mnist1m_sampled_returns_mnist_runner(self, mnist1m_sampled_config):
        assert mnist1m_sampled_config.get_trial_runner_class() is MNISTTrialRunner

    def test_fixed_data_returns_synthetic_fixed_data_runner(self, fixed_data_config):
        assert fixed_data_config.get_trial_runner_class() is SyntheticFixedDataTrialRunner

    def test_linear_teacher_returns_synthetic_fixed_data_runner(self, linear_teacher_config):
        assert linear_teacher_config.get_trial_runner_class() is SyntheticFixedDataTrialRunner

    def test_fixed_time_returns_synthetic_fixed_time_runner(self, fixed_time_config):
        assert fixed_time_config.get_trial_runner_class() is SyntheticFixedTimeTrialRunner

    def test_mlp_teacher_returns_synthetic_fixed_time_runner(self, mlp_teacher_config):
        assert mlp_teacher_config.get_trial_runner_class() is SyntheticFixedTimeTrialRunner

    def test_base_class_raises_not_implemented(self, fixed_time_config):
        with pytest.raises(NotImplementedError):
            ExperimentBase.get_trial_runner_class(fixed_time_config)


class TestPrepareDatasets:
    """Tests for the polymorphic prepare_datasets method on experiment classes."""

    @patch("batch_size_studies.experiments.load_datasets")
    def test_mnist_loads_standard_mnist(self, mock_loader, mnist_config):
        mock_loader.return_value = (
            (np.zeros((100, 28, 28, 1)), np.zeros(100)),
            (np.zeros((20, 28, 28, 1)), np.zeros(20)),
        )

        train_ds, test_ds = mnist_config.prepare_datasets(init_key=0)

        mock_loader.assert_called_once()
        assert train_ds is not None and test_ds is not None
        assert len(train_ds["image"]) == 100
        assert len(test_ds["image"]) == 20

    @patch("batch_size_studies.experiments.load_mnist1m_dataset")
    def test_mnist1m_sampled_loads_and_samples(self, mock_loader, mnist1m_sampled_config):
        # Use arange to easily check if shuffling occurred
        mock_loader.return_value = (
            (np.arange(100).reshape(100, 1, 1, 1), np.arange(100)),
            (np.zeros((20, 28, 28, 1)), np.zeros(20)),
        )

        # Modify the config to request a smaller subset
        config = replace(mnist1m_sampled_config, max_train_samples=50)

        train_ds, test_ds = config.prepare_datasets(init_key=42)

        mock_loader.assert_called_once()
        assert train_ds is not None and test_ds is not None
        assert len(train_ds["image"]) == 50
        # Check that it's not just the first 50 samples (i.e., it was shuffled)
        assert not np.array_equal(train_ds["image"].flatten(), np.arange(50))

    def test_synthetic_generates_data(self, fixed_data_config):
        # Mock the generate_data method on the instance to check if it's called
        with patch.object(
            SyntheticExperimentFixedData, "generate_data", return_value=(np.zeros((100, 10)), np.zeros(100))
        ) as mock_gen:
            train_ds, test_ds = fixed_data_config.prepare_datasets(init_key=0)

            mock_gen.assert_called_once()
            assert test_ds is None
            assert isinstance(train_ds, tuple) and len(train_ds) == 2

    def test_handles_dataset_not_found(self, mnist_config, caplog):
        # Mock the loader to raise FileNotFoundError
        mock_loader = Mock(side_effect=FileNotFoundError("Dataset file missing"))

        train_ds, test_ds = mnist_config.prepare_datasets(init_key=0, dataset_loader=mock_loader)

        assert train_ds is None and test_ds is None
        assert "Dataset not found" in caplog.text

    def test_no_data_loaded_for_fixed_time(self, fixed_time_config):
        train_ds, test_ds = fixed_time_config.prepare_datasets(init_key=0)

        assert train_ds is None and test_ds is None


class TestSyntheticExperimentMLPTeacher:
    """A test class to group all tests related to the MLP Teacher experiment."""

    @pytest.mark.parametrize(
        "invalid_param, invalid_value, expected_match",
        [
            (
                "teacher_N",
                64.0,
                "Attribute 'teacher_N' expected type int, but got float",
            ),
            ("gamma", 1, "Attribute 'gamma' expected type float, but got int"),
            (
                "parameterization",
                "SP",
                "Attribute 'parameterization' expected type Parameterization",
            ),
        ],
    )
    def test_strict_type_enforcement(self, invalid_param, invalid_value, expected_match):
        """Tests that the strict type checker catches various incorrect types."""
        base_config = {
            "D": 16,
            "P": 128,
            "N": 32,
            "L": 2,
            "gamma": 1.0,
            "parameterization": Parameterization.SP,
            "num_steps": 100,
            "teacher_N": 64,
            "teacher_L": 3,
            "teacher_gamma": 1.0,
            "teacher_parameterization": Parameterization.SP,
            "optimizer": OptimizerType.SGD,
            "loss_type": LossType.MSE,
        }
        base_config[invalid_param] = invalid_value

        with pytest.raises(TypeError, match=expected_match):
            SyntheticExperimentMLPTeacher(**base_config)


class TestExperimentBehavior:
    """Tests general, shared behaviors of experiment classes."""

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "fixed_time_config",
            "fixed_data_config",
            "mlp_teacher_config",
            "linear_teacher_config",
        ],
    )
    def test_teacher_weights_are_deterministic(self, config_fixture, request):
        """Tests that teacher weight generation is deterministic for synthetic experiments."""
        config = request.getfixturevalue(config_fixture)
        weights1 = config.generate_teacher_weights()
        weights2 = config.generate_teacher_weights()
        for w1, w2 in zip(weights1, weights2):
            np.testing.assert_array_equal(w1, w2)

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "fixed_time_config",
            "fixed_data_config",
            "mlp_teacher_config",
            "linear_teacher_config",
        ],
    )
    def test_data_generation_is_deterministic(self, config_fixture, request):
        config = request.getfixturevalue(config_fixture)
        key = jr.key(42)
        X1, y1 = config.generate_data(key)
        X2, y2 = config.generate_data(key)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    @pytest.mark.parametrize(
        "config_fixture",
        [
            "fixed_time_config",
            "fixed_data_config",
            "mlp_teacher_config",
            "linear_teacher_config",
            "mnist_config",
            "mnist1m_config",
            "mnist1m_sampled_config",
        ],
    )
    def test_plot_title_does_not_crash(self, config_fixture, request):
        """Smoke test to ensure plot_title() runs and returns a string."""
        config = request.getfixturevalue(config_fixture)
        title = config.plot_title()
        assert isinstance(title, str)


class TestIsRunComplete:
    """Tests the polymorphic is_run_complete method for all experiment types."""

    @pytest.mark.parametrize(
        "config_fixture, run_key, complete_result, incomplete_result",
        [
            # Fixed Time (e.g., SyntheticExperimentFixedTime, SyntheticExperimentMLPTeacher)
            (
                "fixed_time_config",  # num_steps=100
                RunKey(32, 0.1),
                {"loss_history": [0.1] * 100},
                {"loss_history": [0.1] * 99},
            ),
            # Fixed Data (e.g., SyntheticExperimentFixedData, SyntheticExperimentLinearTeacher)
            (
                "fixed_data_config",  # P=128, num_epochs=1 (default)
                RunKey(32, 0.1),  # steps = 1 * (128//32) = 4
                {"loss_history": [0.1] * 4},
                {"loss_history": [0.1] * 3},
            ),
            # MNIST-based (e.g., MNISTExperiment, MNIST1MExperiment, MNIST1MSampledExperiment)
            (
                "mnist_config",  # num_epochs=4
                RunKey(128, 0.1),
                {"epoch_test_accuracies": [0.9] * 4},
                {"epoch_test_accuracies": [0.9] * 3},
            ),
        ],
    )
    def test_completion_logic(self, config_fixture, run_key, complete_result, incomplete_result, request):
        config = request.getfixturevalue(config_fixture)
        assert config.is_run_complete(complete_result, run_key) is True
        assert config.is_run_complete(incomplete_result, run_key) is False
        # Test with a result dictionary that's missing the relevant key
        assert config.is_run_complete({"other_metric": 1}, run_key) is False


class TestFilenameUniqueness:
    """
    Tests that changing any single parameter that should be part of the filename
    results in a unique filename, preventing collisions.
    """

    # Define base configurations for each experiment type
    base_synthetic_ft = {
        "D": 16,
        "P": 128,
        "N": 32,
        "K": 2,
        "num_steps": 100,
        "gamma": 1.0,
        "L": 2,
        "parameterization": Parameterization.SP,
        "optimizer": OptimizerType.SGD,
        "loss_type": LossType.MSE,
    }
    base_synthetic_fd = {
        "D": 16,
        "P": 128,
        "N": 32,
        "K": 2,
        "gamma": 1.0,
        "L": 2,
        "parameterization": Parameterization.SP,
        "optimizer": OptimizerType.SGD,
        "loss_type": LossType.MSE,
    }
    base_mlp_teacher = {
        "D": 16,
        "P": 128,
        "N": 32,
        "L": 2,
        "gamma": 1.0,
        "parameterization": Parameterization.SP,
        "num_steps": 100,
        "teacher_N": 64,
        "teacher_L": 3,
        "teacher_gamma": 1.0,
        "teacher_parameterization": Parameterization.SP,
        "optimizer": OptimizerType.SGD,
        "loss_type": LossType.MSE,
    }
    base_mnist = {
        "N": 32,
        "L": 2,
        "parameterization": Parameterization.SP,
        "optimizer": OptimizerType.SGD,
        "loss_type": LossType.XENT,
        "gamma": 1.0,
        "num_epochs": 4,
    }
    base_mnist1m = {
        "N": 32,
        "L": 2,
        "parameterization": Parameterization.SP,
        "num_epochs": 5,
        "optimizer": OptimizerType.SGD,
        "loss_type": LossType.XENT,
        "gamma": 1.0,
    }

    # Define the parameters to test for each experiment type
    # Format: (ExperimentClass, base_config, param_name, modified_value)
    test_cases = [
        # SyntheticExperimentFixedTime
        (SyntheticExperimentFixedTime, base_synthetic_ft, "D", 32),
        (SyntheticExperimentFixedTime, base_synthetic_ft, "P", 256),
        (SyntheticExperimentFixedTime, base_synthetic_ft, "N", 64),
        (SyntheticExperimentFixedTime, base_synthetic_ft, "K", 3),
        (SyntheticExperimentFixedTime, base_synthetic_ft, "gamma", 2.0),
        (SyntheticExperimentFixedTime, base_synthetic_ft, "L", 3),
        (
            SyntheticExperimentFixedTime,
            base_synthetic_ft,
            "parameterization",
            Parameterization.MUP,
        ),
        (
            SyntheticExperimentFixedTime,
            base_synthetic_ft,
            "optimizer",
            OptimizerType.ADAM,
        ),
        # SyntheticExperimentFixedData
        (SyntheticExperimentFixedData, base_synthetic_fd, "D", 32),
        (SyntheticExperimentFixedData, base_synthetic_fd, "N", 64),
        (SyntheticExperimentFixedData, base_synthetic_fd, "gamma", 2.0),
        # SyntheticExperimentMLPTeacher
        (SyntheticExperimentMLPTeacher, base_mlp_teacher, "teacher_N", 128),
        (SyntheticExperimentMLPTeacher, base_mlp_teacher, "teacher_L", 4),
        (SyntheticExperimentMLPTeacher, base_mlp_teacher, "gamma", 0.5),
        # MNISTExperiment
        (MNISTExperiment, base_mnist, "N", 64),
        (MNISTExperiment, base_mnist, "L", 3),
        (MNISTExperiment, base_mnist, "parameterization", Parameterization.MUP),
        (MNISTExperiment, base_mnist, "optimizer", OptimizerType.ADAM),
        (MNISTExperiment, base_mnist, "loss_type", LossType.MSE),
        (MNISTExperiment, base_mnist, "gamma", 2.0),
        # MNIST1MExperiment
        (MNIST1MExperiment, base_mnist1m, "N", 64),
        (MNIST1MExperiment, base_mnist1m, "L", 3),
        (MNIST1MExperiment, base_mnist1m, "parameterization", Parameterization.MUP),
        (MNIST1MExperiment, base_mnist1m, "optimizer", OptimizerType.ADAM),
        (MNIST1MExperiment, base_mnist1m, "loss_type", LossType.MSE),
        (MNIST1MExperiment, base_mnist1m, "gamma", 2.0),
    ]

    @pytest.mark.parametrize("exp_class, base_config, param, new_value", test_cases)
    def test_filename_is_unique_per_parameter(self, exp_class, base_config, param, new_value):
        """
        Verifies that changing a single parameter results in a unique filename,
        ensuring no accidental collisions.
        """
        # Create the base experiment instance
        base_exp = exp_class(**base_config)

        # Create the modified experiment instance
        modified_config = base_config.copy()
        modified_config[param] = new_value
        modified_exp = exp_class(**modified_config)

        # Generate filenames and assert they are different
        base_filename = base_exp.generate_filename()
        modified_filename = modified_exp.generate_filename()

        assert base_filename != modified_filename, (
            f"Changing '{param}' did not produce a unique filename for {exp_class.__name__}"
        )
