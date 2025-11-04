import logging
import os
from dataclasses import replace
from types import SimpleNamespace

import jax.random as jr
import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import (
    MNIST1MExperiment,
    MNIST1MSampledExperiment,
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
    SyntheticExperimentLinearTeacher,
    SyntheticExperimentMLPTeacher,
    _load_mnist_dataset,
    _subsample_mnist_data,
)
from batch_size_studies.storage_utils import generate_experiment_filename, save_experiment

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


class TestExperimentBehavior:


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


class TestFilenameUniqueness:





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


class TestLegacyStorageCompatibility:
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
    def test_legacy_results_and_weights_are_discoverable(self, tmp_path, config_fixture, request):
        experiment = request.getfixturevalue(config_fixture)

        experiments_root = tmp_path / "experiments" / experiment.experiment_type
        experiments_root.mkdir(parents=True, exist_ok=True)

        params = experiment.to_params_dict()
        legacy_params = dict(params)
        legacy_params.pop("loss_type", None)

        legacy_results_filename = generate_experiment_filename(legacy_params, prefix="results", extension="pkl")
        legacy_results_path = experiments_root / legacy_results_filename
        legacy_results_payload = {"losses": {"legacy": {"loss_history": [1.0]}}, "failed_runs": set()}
        save_experiment(legacy_results_payload, str(legacy_results_path))

        legacy_base = os.path.splitext(generate_experiment_filename(legacy_params, prefix="", extension="pkl"))[0]
        legacy_weights_path = experiments_root / f"{legacy_base}_weights.pkl"
        legacy_weights_payload = {"initial_params": {"theta": 0.1}, "weight_snapshots": {}}
        save_experiment(legacy_weights_payload, str(legacy_weights_path))

        experiments_dir = tmp_path / "experiments"
        manager = CheckpointManager(experiment, directory=str(experiments_dir))
        assert manager.weights_filepath == str(legacy_weights_path)
        assert manager.checkpoint_dir == str(experiments_root / f"{legacy_base}_checkpoints")
        assert manager.load_initial_params() == legacy_weights_payload["initial_params"]

        loaded_losses, loaded_failed = experiment.load_results(directory=str(experiments_dir))
        assert loaded_losses == legacy_results_payload["losses"]
        assert loaded_failed == legacy_results_payload["failed_runs"]


class TestExperimentUtilities:
    def test_to_params_dict_converts_enums_and_excludes_fixed_fields(self, mnist_config):
        params = mnist_config.to_params_dict()
        assert params["optimizer"] == mnist_config.optimizer.value
        assert params["loss_type"] == mnist_config.loss_type.value
        assert "num_outputs" not in params
        assert "experiment_type" not in params

    def test_filename_variants_include_legacy(self, mnist_config):
        variants = mnist_config.get_filename_variants()
        assert variants  # non-empty
        assert any("loss_type" in name for name in variants)
        assert any("loss_type" not in name for name in variants)

    def test_save_and_load_results_roundtrip(self, tmp_path, mnist_config, caplog):
        run_key = RunKey(batch_size=1, eta=0.5)
        losses = {run_key: {"loss_history": [0.1], "expected_steps": 1}}
        failed = set()

        with caplog.at_level(logging.INFO):
            filepath = mnist_config.save_results(losses, failed, directory=str(tmp_path))
            assert os.path.exists(filepath)
            loaded_losses, loaded_failed = mnist_config.load_results(directory=str(tmp_path), silent=False)

        assert "Results file found" in "".join(caplog.messages)
        assert loaded_losses == losses
        assert loaded_failed == failed

    def test_mnist_prepare_datasets_success(self, monkeypatch, mnist_config):
        dummy_train = (np.ones((4, 28, 28)), np.zeros(4, dtype=int))
        dummy_test = (np.zeros((2, 28, 28)), np.ones(2, dtype=int))

        def fake_loader():
            return dummy_train, dummy_test

        monkeypatch.setattr("batch_size_studies.experiments.load_datasets", fake_loader)
        train_ds, test_ds = mnist_config.prepare_datasets(init_key=0)

        np.testing.assert_array_equal(train_ds["image"], dummy_train[0])
        np.testing.assert_array_equal(test_ds["label"], dummy_test[1])

    def test_load_mnist_dataset_handles_failure(self, caplog, mnist_config):
        def failing_loader():
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR):
            train_ds, test_ds = _load_mnist_dataset(mnist_config, init_key=0, dataset_loader=failing_loader)

        assert train_ds is None and test_ds is None
        assert "Failed to load dataset" in "".join(caplog.messages)

    def test_subsample_mnist_data_respects_max_samples(self, mnist1m_sampled_config):
        images = np.arange(10 * 3).reshape(10, 3)
        labels = np.arange(10)
        experiment = SimpleNamespace(max_train_samples=4)

        sub_images, sub_labels = _subsample_mnist_data(images, labels, experiment, init_key=0)

        assert len(sub_images) == 4
        assert len(sub_labels) == 4

    def test_adjusted_eta_scaling(self, mnist_config):
        base_eta = 0.01

        # SP parameterization should leave eta unchanged when gamma = 1
        assert mnist_config.get_adjusted_eta(base_eta) == pytest.approx(base_eta)

        mup_sgd = replace(mnist_config, parameterization=Parameterization.MUP, optimizer=OptimizerType.SGD)
        assert mup_sgd.get_adjusted_eta(base_eta) == pytest.approx(base_eta * mup_sgd.N)

        mup_adam = replace(mup_sgd, optimizer=OptimizerType.ADAM)
        assert mup_adam.get_adjusted_eta(base_eta) == pytest.approx(base_eta * np.sqrt(mup_adam.N))

    def test_mnist1m_sampled_prepare_datasets_passes_forced_seed(self, monkeypatch, mnist1m_sampled_config):
        captured = {}

        def fake_loader(experiment, init_key, dataset_loader=None, forced_subsample_seed=None):
            captured["experiment"] = experiment
            captured["forced"] = forced_subsample_seed
            dummy = {"image": np.zeros((1, 28, 28)), "label": np.zeros(1, dtype=int)}
            return dummy, dummy

        monkeypatch.setattr("batch_size_studies.experiments._load_mnist_dataset", fake_loader)
        train_ds, test_ds = mnist1m_sampled_config.prepare_datasets(init_key=123, forced_subsample_seed=999)

        assert captured["experiment"] is mnist1m_sampled_config
        assert captured["forced"] == 999
        assert train_ds["image"].shape == (1, 28, 28)
        assert test_ds["label"].shape == (1,)

    def test_mnist1m_sampled_metadata_and_skip_logic(self, mnist1m_sampled_config):
        assert mnist1m_sampled_config.get_sweep_metadata(7) == {"subsample_seed": 7}

        train_ds = {"image": np.zeros((50, 28, 28)), "label": np.zeros(50, dtype=int)}
        assert not mnist1m_sampled_config.should_skip_batch_size(32, train_ds)
        assert mnist1m_sampled_config.should_skip_batch_size(1024, train_ds)

        num_steps, epochs = mnist1m_sampled_config.compute_num_steps(batch_size=50, train_ds=train_ds, num_epochs=None)
        assert num_steps == mnist1m_sampled_config.num_epochs * (len(train_ds["image"]) // 50)
        assert epochs == mnist1m_sampled_config.num_epochs

    def test_synthetic_experiment_behaviors(self, fixed_time_config, fixed_data_config):
        assert fixed_time_config.is_online_experiment()
        assert not fixed_time_config.should_skip_batch_size(16)
        X_online, y_online = fixed_time_config.generate_data(jr.key(0))
        assert X_online.shape[0] == fixed_time_config.P
        assert y_online.shape[0] == fixed_time_config.P

        train_ds = {"image": np.zeros((fixed_data_config.P, 1)), "label": np.zeros(fixed_data_config.P)}
        assert not fixed_data_config.should_skip_batch_size(32, train_ds)
        assert fixed_data_config.should_skip_batch_size(fixed_data_config.P + 1, train_ds)

        num_steps, epochs = fixed_data_config.compute_num_steps(batch_size=32, train_ds=train_ds, num_epochs=None)
        assert epochs == fixed_data_config.num_epochs
        assert num_steps == epochs * (fixed_data_config.P // 32)
