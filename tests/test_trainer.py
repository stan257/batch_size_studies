import logging
import os
from dataclasses import dataclass, replace

import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import (
    ExperimentBase,
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
    SyntheticExperimentLinearTeacher,
)
from batch_size_studies.runner import MNISTTrialRunner, SyntheticFixedDataTrialRunner, run_experiment_sweep

# --- Fixtures ---


@pytest.fixture
def fixed_time_config():
    """Fixture for a fast-to-run FixedTime experiment."""
    return SyntheticExperimentFixedTime(
        D=8,
        P=32,
        N=16,
        K=2,
        num_steps=10,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )


@pytest.fixture
def fixed_data_config():
    """Fixture for a fast-to-run FixedData experiment."""
    return SyntheticExperimentFixedData(
        D=8,
        P=32,
        N=16,
        K=2,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )


@pytest.fixture
def linear_teacher_config_subset():
    """Fixture for a Linear Teacher experiment with non-divisible data size."""
    return SyntheticExperimentLinearTeacher(
        D=8,
        P=105,  # Not divisible by common batch sizes
        alpha=1.0,
        beta=1.0,
        num_epochs=2,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )


@pytest.fixture
def mnist_config():
    """Fixture for a fast-to-run MNIST experiment."""
    return MNISTExperiment(
        N=32,
        L=2,
        num_epochs=4,
        parameterization=Parameterization.SP,
        gamma=1.0,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.XENT,
    )


@pytest.fixture
def mnist_config_subset():
    """Fixture for an MNIST experiment for subset testing."""
    return MNISTExperiment(
        N=32,
        L=2,
        num_epochs=2,
        parameterization=Parameterization.SP,
        gamma=1.0,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.XENT,
    )


@pytest.fixture
def mock_mnist_loader():
    """A mock dataset loader that returns small numpy arrays."""

    def _loader():
        np.random.seed(42)
        train_images = np.random.rand(128, 28, 28, 1).astype(np.float32)
        train_labels = np.random.randint(0, 10, 128).astype(np.int32)
        test_images = np.random.rand(64, 28, 28, 1).astype(np.float32)
        test_labels = np.random.randint(0, 10, 64).astype(np.int32)
        return (train_images, train_labels), (test_images, test_labels)

    return _loader


# --- Test Classes ---


class TestUnifiedRunner:
    def test_handles_unknown_experiment_type(self, tmp_path, caplog):
        """Tests that the runner handles an unknown experiment type gracefully."""

        @dataclass(frozen=True)
        class UnknownExperiment(ExperimentBase):
            experiment_type: str = "unknown"

            def is_run_complete(self, result, run_key):
                return False

            def should_skip_batch_size(self, batch_size, train_ds_size=None):
                return False

        config = UnknownExperiment(optimizer=OptimizerType.SGD, loss_type=LossType.MSE)
        # The runner should fail fast if it doesn't know what model to use.
        with pytest.raises(TypeError, match="Unknown student model for experiment type: UnknownExperiment"):
            losses, failures = run_experiment_sweep(
                experiment=config, batch_sizes=[8], etas=[0.1], directory=str(tmp_path)
            )


class TestSyntheticRunner:
    def test_runs_and_returns_correct_structure(self, fixed_time_config, tmp_path):
        """Tests that the main training function runs without error and returns the expected data structures."""
        losses, failures = run_experiment_sweep(
            experiment=fixed_time_config, batch_sizes=[4, 8], etas=[0.1], directory=str(tmp_path)
        )
        assert isinstance(losses, dict)
        assert isinstance(failures, set)
        assert len(losses) == 2
        assert len(failures) == 0
        expected_key = RunKey(batch_size=4, eta=0.1)
        assert "loss_history" in losses[expected_key]
        assert len(losses[expected_key]["loss_history"]) == fixed_time_config.num_steps

    def test_handles_failed_runs(self, fixed_time_config, tmp_path):
        """Tests that the training function correctly identifies and logs runs that fail with NaN/inf losses."""
        losses, failures = run_experiment_sweep(
            experiment=fixed_time_config, batch_sizes=[4], etas=[1e6], directory=str(tmp_path)
        )
        assert len(losses) == 0
        assert failures == {RunKey(batch_size=4, eta=1e6)}

    def test_run_is_reproducible(self, fixed_time_config, tmp_path):
        """Tests that two identical training runs produce the exact same results."""
        losses1, failed1 = run_experiment_sweep(
            experiment=fixed_time_config,
            batch_sizes=[4, 8],
            etas=[0.1, 0.01],
            init_key=42,
            directory=str(tmp_path / "run1"),
        )
        losses2, failed2 = run_experiment_sweep(
            experiment=fixed_time_config,
            batch_sizes=[4, 8],
            etas=[0.1, 0.01],
            init_key=42,
            directory=str(tmp_path / "run2"),
        )
        assert failed1 == failed2
        assert losses1.keys() == losses2.keys()
        for key in losses1:
            np.testing.assert_allclose(losses1[key]["loss_history"], losses2[key]["loss_history"])

    def test_run_with_fixed_data(self, fixed_data_config, tmp_path):
        num_epochs, batch_size = 3, 8
        expected_steps = num_epochs * (fixed_data_config.P // batch_size)
        losses, _ = run_experiment_sweep(
            experiment=fixed_data_config,
            batch_sizes=[batch_size],
            etas=[0.1],
            init_key=0,
            num_epochs=num_epochs,
            directory=str(tmp_path),
        )
        expected_key = RunKey(batch_size=batch_size, eta=0.1)
        assert "loss_history" in losses[expected_key]
        assert len(losses[expected_key]["loss_history"]) == expected_steps

    def test_skips_run_if_batch_size_exceeds_p_for_fixed_data(self, fixed_data_config, caplog, tmp_path):
        with caplog.at_level(logging.WARNING):
            losses, failures = run_experiment_sweep(
                experiment=fixed_data_config, batch_sizes=[16, 64], etas=[0.1], num_epochs=1, directory=str(tmp_path)
            )
        assert RunKey(16, 0.1) in losses
        assert RunKey(64, 0.1) not in losses
        assert RunKey(64, 0.1) not in failures
        assert "Skipping batch size 64 > dataset size P (32)" in caplog.text

    def test_sweep_runs_all_combinations_by_default(self, fixed_data_config, tmp_path):
        """
        Tests that without eta_stability_search_depth, the sweep runs all combinations.
        """
        batch_sizes = [8, 16]
        etas = [0.1, 0.01]

        # Use a config that is known to converge and has enough data
        converging_config = replace(fixed_data_config, P=32)

        results, failures = run_experiment_sweep(
            experiment=converging_config,
            batch_sizes=batch_sizes,
            etas=etas,
            num_epochs=1,
            directory=str(tmp_path),
            # NOTE: eta_stability_search_depth is intentionally omitted
        )

        # Check that the number of successful runs matches the total number of combinations
        assert len(results) == len(batch_sizes) * len(etas)
        assert not failures

        # Verify that every specific run key is present in the results
        for bs in batch_sizes:
            for eta in etas:
                assert RunKey(bs, eta) in results

    def test_eta_stability_search_stops_early(self, fixed_data_config, tmp_path, monkeypatch):
        """
        Tests that the eta stability search correctly stops after a consecutive number of successes,
        and that the counter resets upon failure.
        """
        batch_sizes = [16]
        # Etas are sorted descending by the runner
        etas = [1.0, 0.5, 0.25, 0.125, 0.06]
        # Define which etas will "converge". 0.5 will fail, resetting the counter.
        converging_etas = {1.0, 0.25, 0.125, 0.06}
        eta_stability_search_depth = 2

        # Keep track of which etas were actually run
        run_etas = []

        def mock_run(self):
            run_etas.append(self.run_key.eta)
            if self.run_key.eta in converging_etas:
                return {"loss_history": [0.5, 0.4]}  # Minimal success result
            return None  # Failure result

        # Patch the trial runner's run method to control convergence
        monkeypatch.setattr("batch_size_studies.trainer.SyntheticFixedDataTrialRunner.run", mock_run)

        results, failures = run_experiment_sweep(
            experiment=fixed_data_config,
            batch_sizes=batch_sizes,
            etas=etas,
            num_epochs=1,
            directory=str(tmp_path),
            eta_stability_search_depth=eta_stability_search_depth,
        )

        # The sweep should proceed as follows for B=16:
        # eta=1.0:   Converges. consecutive_successes = 1.
        # eta=0.5:   Fails.     consecutive_successes = 0. (Counter reset)
        # eta=0.25:  Converges. consecutive_successes = 1.
        # eta=0.125: Converges. consecutive_successes = 2. -> STOP.
        # eta=0.06:  Should not be run.
        expected_run_etas = [1.0, 0.5, 0.25, 0.125]
        assert run_etas == expected_run_etas, "The sweep did not run the expected sequence of etas."
        assert RunKey(16, 0.06) not in results and RunKey(16, 0.06) not in failures


class TestMNISTRunner:
    def test_runs_and_returns_correct_structure(self, mnist_config, mock_mnist_loader, tmp_path):
        """Tests that the MNIST runner completes and returns the correct structure."""
        results, failures = run_experiment_sweep(
            experiment=mnist_config,
            batch_sizes=[32],
            etas=[0.01],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path),
        )
        assert isinstance(results, dict) and isinstance(failures, set)
        assert len(failures) == 0 and len(results) == 1
        run_key = RunKey(batch_size=32, eta=0.01)
        assert run_key in results
        assert "final_test_accuracy" in results[run_key]
        assert len(results[run_key]["epoch_test_accuracies"]) == mnist_config.num_epochs

    def test_checkpoint_and_resume(self, mnist_config, mock_mnist_loader, tmp_path, caplog):
        """Tests that an interrupted MNIST experiment correctly resumes from the last completed epoch."""
        total_epochs, resume_from_epoch = mnist_config.num_epochs, 2
        run_key = RunKey(batch_size=64, eta=0.01)

        # Run partway
        run_experiment_sweep(
            experiment=mnist_config,
            batch_sizes=[64],
            etas=[0.01],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path),
            num_epochs=resume_from_epoch,
        )
        cm = CheckpointManager(mnist_config, directory=str(tmp_path))
        resume_file = cm._get_resume_filepath(run_key)
        assert os.path.exists(resume_file)

        # Run to completion
        caplog.clear()
        with caplog.at_level(logging.INFO):
            results, _ = run_experiment_sweep(
                experiment=mnist_config,
                batch_sizes=[64],
                etas=[0.01],
                dataset_loader=mock_mnist_loader,
                directory=str(tmp_path),
                num_epochs=total_epochs,
            )

        assert f"Resuming run {run_key} from step {resume_from_epoch}" in caplog.text
        assert len(results[run_key]["epoch_test_accuracies"]) == total_epochs
        assert not os.path.exists(resume_file)

    def test_handles_failed_runs(self, mnist_config, mock_mnist_loader, tmp_path):
        """Tests that a run that diverges is correctly marked as failed."""
        _, failures = run_experiment_sweep(
            experiment=mnist_config,
            batch_sizes=[32],
            etas=[1e20],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path),
        )
        assert len(failures) == 1 and RunKey(batch_size=32, eta=1e20) in failures

    def test_optimizer_selection_works(self, mnist_config, mock_mnist_loader, tmp_path):
        """Tests that changing the optimizer in the config results in a different training outcome."""
        from dataclasses import replace

        import jax

        run_key = RunKey(batch_size=64, eta=0.1)
        last_epoch = mnist_config.num_epochs - 1

        # Run with SGD
        sgd_config = mnist_config
        run_experiment_sweep(
            experiment=sgd_config,
            batch_sizes=[64],
            etas=[0.1],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path / "sgd"),
        )
        cm_sgd = CheckpointManager(sgd_config, directory=str(tmp_path / "sgd"))
        params_sgd = cm_sgd.load_analysis_snapshot(run_key, step=last_epoch)

        # Run with Adam
        adam_config = replace(mnist_config, optimizer=OptimizerType.ADAM)
        run_experiment_sweep(
            experiment=adam_config,
            batch_sizes=[64],
            etas=[0.1],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path / "adam"),
        )
        cm_adam = CheckpointManager(adam_config, directory=str(tmp_path / "adam"))
        params_adam = cm_adam.load_analysis_snapshot(run_key, step=last_epoch)

        assert params_sgd is not None and params_adam is not None
        sgd_leaves, _ = jax.tree_util.tree_flatten(params_sgd)
        adam_leaves, _ = jax.tree_util.tree_flatten(params_adam)
        are_different = any(not np.allclose(s, a) for s, a in zip(sgd_leaves, adam_leaves))
        assert are_different, "Final model parameters for SGD and Adam were unexpectedly identical."


class TestEpochBasedDataHandling:
    """Tests for data handling in epoch-based runners."""

    @pytest.mark.parametrize(
        "runner_class, config_fixture, data_setup",
        [
            (
                SyntheticFixedDataTrialRunner,
                "linear_teacher_config_subset",
                # For synthetic, data is passed as X_data, y_data
                lambda config: {
                    "X_data": np.arange(config.P).reshape(-1, 1),
                    "y_data": np.zeros(config.P),
                },
            ),
            (
                MNISTTrialRunner,
                "mnist_config_subset",
                # For MNIST, data is passed as a dict. We use 105 samples.
                lambda config: {
                    "train_ds": {
                        "image": np.arange(105).reshape(105, 1),
                        "label": np.zeros(105),
                    },
                    "test_ds": {"image": np.array([]), "label": np.array([])},
                },
            ),
        ],
    )
    def test_data_subset_is_consistent_across_epochs(self, runner_class, config_fixture, data_setup, request):
        """
        Verifies that for multi-epoch runs, the exact same subset of data is
        used in each epoch, but that the order is shuffled differently.
        """
        config = request.getfixturevalue(config_fixture)
        batch_size = 20  # 105 is not divisible by 20
        run_key = RunKey(batch_size=batch_size, eta=0.1)

        # --- Setup runner with mock data ---
        # The data arrays will contain indices instead of actual data
        # to make it easy to track which samples are used.
        data_kwargs = data_setup(config)
        num_samples = 105

        # Common kwargs for both runners
        runner_kwargs = {
            "experiment": config,
            "run_key": run_key,
            "params0": None,
            "model_instance": None,
            "checkpoint_manager": None,
            "pbar": None,
            "no_save": True,
            "init_key": 0,
            "num_epochs": 2,
            **data_kwargs,
        }
        runner = runner_class(**runner_kwargs)

        # --- Collect indices from two epochs ---
        all_indices_by_epoch = []
        for epoch in range(2):
            batch_generator = runner._get_batch_generator(epoch)
            epoch_indices = []
            for x_batch, _ in batch_generator:
                # The generator yields (x, y), we only care about x.
                # Flatten to get the original indices back.
                epoch_indices.extend(x_batch.flatten())
            all_indices_by_epoch.append(np.array(epoch_indices))

        # --- Assertions ---
        indices_epoch0 = all_indices_by_epoch[0]
        indices_epoch1 = all_indices_by_epoch[1]

        # 1. Check that the number of samples is correct (truncated)
        num_usable_samples = (num_samples // batch_size) * batch_size  # 5 * 20 = 100
        assert len(indices_epoch0) == num_usable_samples
        assert len(indices_epoch1) == num_usable_samples

        # 2. Check that the set of indices used is identical between epochs
        sorted_indices_epoch0 = np.sort(indices_epoch0)
        sorted_indices_epoch1 = np.sort(indices_epoch1)
        np.testing.assert_array_equal(
            sorted_indices_epoch0,
            sorted_indices_epoch1,
            err_msg="The set of data samples should be identical across epochs.",
        )

        # 3. Check that the order of indices is different between epochs (shuffling works)
        assert not np.array_equal(indices_epoch0, indices_epoch1), (
            "The order of data samples should be different in each epoch."
        )
