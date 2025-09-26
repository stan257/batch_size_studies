import jax.random as jr
import numpy as np
import pytest

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization
from batch_size_studies.experiments import (
    MNIST1MExperiment,
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
    SyntheticExperimentMLPTeacher,
)


# --- Tests for SyntheticExperimentFixedTime ---
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
    )


class TestSyntheticExperimentFixedTime:
    """Groups tests for the SyntheticExperimentFixedTime class."""

    def test_initialization(self, fixed_time_config):
        assert fixed_time_config.K == 2
        assert fixed_time_config.experiment_type == "fixed_time_poly_teacher"

    def test_teacher_weights_are_deterministic(self, fixed_time_config):
        weights1 = fixed_time_config.generate_teacher_weights()
        weights2 = fixed_time_config.generate_teacher_weights()
        assert weights1.shape == (16, 1)
        np.testing.assert_array_equal(weights1, weights2)

    def test_data_generation_is_deterministic(self, fixed_time_config):
        key = jr.key(42)
        X1, y1 = fixed_time_config.generate_data(key)
        X2, y2 = fixed_time_config.generate_data(key)
        assert X1.shape == (128, 16)
        assert y1.shape == (128, 1)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_plot_title(self, fixed_time_config):
        title = fixed_time_config.plot_title()
        assert "T* = 100" in title
        assert "poly task" in title


# --- Tests for SyntheticExperimentFixedData ---
@pytest.fixture
def fixed_data_config():
    """Fixture for a standard FixedData experiment configuration."""
    return SyntheticExperimentFixedData(D=16, P=128, N=32, K=2, gamma=1.0, L=2, parameterization=Parameterization.SP)


class TestSyntheticExperimentFixedData:
    """Groups tests for the SyntheticExperimentFixedData class."""

    def test_initialization(self, fixed_data_config):
        assert fixed_data_config.K == 2
        assert fixed_data_config.experiment_type == "fixed_data_poly_teacher"

    def test_filename_is_deterministic(self, fixed_data_config):
        exp2 = SyntheticExperimentFixedData(
            D=16, P=128, N=32, K=2, gamma=1.0, L=2, parameterization=Parameterization.SP
        )
        assert fixed_data_config.generate_filename() == exp2.generate_filename()

    def test_plot_title(self, fixed_data_config):
        title = fixed_data_config.plot_title()
        assert "P = 128" in title
        assert "poly task" in title


# --- Tests for SyntheticExperimentMLPTeacher ---
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
    )


class TestSyntheticExperimentMLPTeacher:
    """A test class to group all tests related to the MLP Teacher experiment."""

    def test_initialization(self, mlp_teacher_config):
        """Tests that the MLP teacher experiment initializes correctly."""
        assert mlp_teacher_config.D == 16
        assert mlp_teacher_config.teacher_N == 64
        assert mlp_teacher_config.experiment_type == "fixed_time_mlp_teacher"

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
        }
        base_config[invalid_param] = invalid_value

        with pytest.raises(TypeError, match=expected_match):
            SyntheticExperimentMLPTeacher(**base_config)

    def test_teacher_weights_are_deterministic(self, mlp_teacher_config):
        """
        Tests that the generated teacher MLP weights are deterministic and have the correct structure.
        """
        weights1 = mlp_teacher_config.generate_teacher_weights()
        weights2 = mlp_teacher_config.generate_teacher_weights()

        assert isinstance(weights1, list)
        assert len(weights1) == mlp_teacher_config.teacher_L
        assert weights1[0].shape == (16, 64)
        assert weights1[1].shape == (64, 64)
        assert weights1[2].shape == (64, 1)

        for w1, w2 in zip(weights1, weights2):
            np.testing.assert_array_equal(w1, w2)

    def test_data_generation_is_deterministic(self, mlp_teacher_config):
        """
        Tests that the data generated by the MLP teacher is deterministic for a given key.
        """
        key = jr.key(42)
        X1, y1 = mlp_teacher_config.generate_data(key)
        X2, y2 = mlp_teacher_config.generate_data(key)

        assert X1.shape == (mlp_teacher_config.P, mlp_teacher_config.D)
        assert y1.shape == (mlp_teacher_config.P, 1)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_filename_is_correct(self, mlp_teacher_config):
        """
        Tests that the filename includes teacher-specific parameters and is clean.
        """
        filename = mlp_teacher_config.generate_filename()

        assert "teacher_N=64" in filename
        assert "teacher_parameterization=SP" in filename
        assert "experiment_type" not in filename

    def test_plot_title(self, mlp_teacher_config):
        """Tests that the plot title is generated correctly."""
        title = mlp_teacher_config.plot_title()
        assert "T* = 100" in title
        assert "MLP teacher" in title
        assert "T(N=64, L=3)" in title
        assert "in SP w/ $N=32, L=2, \\gamma=1.0$" in title


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
