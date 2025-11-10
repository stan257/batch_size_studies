import logging
import os
from dataclasses import dataclass, replace

import jax
import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import (
    CheckpointManager,
    load_final_weights_for_experiment,
)
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import (
    ExperimentBase,
    LinearStudentExperiment,
    MNISTExperiment,
    SyntheticExperimentFixedData,
    SyntheticExperimentFixedTime,
)
from batch_size_studies.runner import run_experiment_sweep
from batch_size_studies.trainer import MNISTTrialRunner

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


class TestSweepRunnerIntegration:
    def test_handles_unknown_experiment_type(self, tmp_path, caplog):


        @dataclass(frozen=True)
        class UnknownExperiment(LinearStudentExperiment, ExperimentBase):
            experiment_type: str = "unknown"
            D: int = 10

            def is_run_complete(self, result, run_key):
                return False

            def should_skip_batch_size(self, batch_size, train_ds=None):
                return False

            def get_trial_runner_class(self):
                raise NotImplementedError

            def prepare_datasets(self, init_key: int, **kwargs):
                # Return a non-None train_ds to pass the data loading check.
                return (np.array([]), np.array([])), None

            def get_adjusted_eta(self, base_eta: float) -> float:
                return base_eta

            def get_model_widths(self) -> list[int]:
                return [10, 1]

            def compute_num_steps(self, batch_size, train_ds, num_epochs):
                return 10, 1  # Return a tuple (num_steps, num_epochs)

        config = UnknownExperiment(optimizer=OptimizerType.SGD, loss_type=LossType.MSE, D=10)
        # The runner should log an error and return None if get_trial_runner_class fails.
        with caplog.at_level(logging.ERROR):
            losses, failures = run_experiment_sweep(
                experiment=config, batch_sizes=[8], etas=[0.1], directory=str(tmp_path)
            )
        assert "does not implement get_trial_runner_class()" in caplog.text

    def test_synthetic_fixed_time_persists_eval_loss(self, tmp_path):
        experiment = SyntheticExperimentFixedTime(
            D=4, P=16, N=4, K=2, num_steps=5, L=2, gamma=1.0,
            parameterization=Parameterization.SP, optimizer=OptimizerType.SGD, loss_type=LossType.MSE
        )
        losses, failures = run_experiment_sweep(
            experiment=experiment, batch_sizes=[4], etas=[0.01], init_key=0, directory=str(tmp_path)
        )
        assert not failures
        for result in losses.values():
            assert "final_eval_loss" in result
            assert np.isfinite(result["final_eval_loss"])

    def test_runs_and_returns_correct_structure(self, fixed_time_config, tmp_path):

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

        losses, failures = run_experiment_sweep(
            experiment=fixed_time_config, batch_sizes=[4], etas=[1e6], directory=str(tmp_path)
        )
        assert len(losses) == 0
        assert failures == {RunKey(batch_size=4, eta=1e6)}

    def test_mnist_eval_subsampling(self, mnist_config, mock_mnist_loader, tmp_path, monkeypatch):


        recorded_sizes = []
        original_hook = MNISTTrialRunner._post_epoch_hook

        def patched_post_epoch_hook(self, epoch, params, results):
            recorded_sizes.append(self.test_ds["image"].shape[0])
            return original_hook(self, epoch, params, results)

        monkeypatch.setattr(MNISTTrialRunner, "_post_epoch_hook", patched_post_epoch_hook)

        run_experiment_sweep(
            experiment=mnist_config,
            batch_sizes=[32],
            etas=[0.1],
            init_key=0,
            directory=str(tmp_path),
            no_save=True,
            dataset_loader=mock_mnist_loader,
            max_eval_samples=20,
        )

        assert recorded_sizes
        assert all(size == 20 for size in recorded_sizes)

    def test_run_is_reproducible(self, fixed_time_config, tmp_path):

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

    def test_load_final_weights_after_sweep(self, fixed_data_config, tmp_path):
        batch_sizes = [4]
        etas = [0.1, 0.01]
        losses, failures = run_experiment_sweep(
            experiment=fixed_data_config,
            batch_sizes=batch_sizes,
            etas=etas,
            init_key=0,
            directory=str(tmp_path),
        )
        assert not failures
        expected_keys = {RunKey(b, e) for b in batch_sizes for e in etas}
        assert set(losses.keys()) == expected_keys

        final_weights = load_final_weights_for_experiment(fixed_data_config, directory=str(tmp_path))
        assert set(final_weights.keys()) == expected_keys

        manager = CheckpointManager(fixed_data_config, directory=str(tmp_path))
        for run_key, final_params in final_weights.items():
            history = manager.load_full_weight_history(run_key)
            assert history, f"No history recorded for {run_key}"
            final_step = max(history.keys())
            assert_allclose_trees(final_params, history[final_step])


def assert_allclose_trees(a, b, rtol=1e-5, atol=1e-8):
    a_flat, a_tree = jax.tree_util.tree_flatten(a)
    b_flat, b_tree = jax.tree_util.tree_flatten(b)
    assert a_tree == b_tree, "PyTree structures do not match"
    for arr_a, arr_b in zip(a_flat, b_flat):
        np.testing.assert_allclose(arr_a, arr_b, rtol=rtol, atol=atol)

    def test_run_with_fixed_data(self, fixed_data_config, tmp_path):

        num_epochs, batch_size = 3, 8
        expected_steps, _ = fixed_data_config.compute_num_steps(batch_size, None, num_epochs)
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
                return {"loss_history": [0.5] * self.num_steps, "expected_steps": self.num_steps}
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

    def test_mnist_checkpoint_and_resume(self, mnist_config, mock_mnist_loader, tmp_path, caplog, monkeypatch):

        total_epochs = mnist_config.num_epochs  # 4
        run_key = RunKey(batch_size=64, eta=0.01)
        steps_per_epoch = 128 // 64  # 2

        # --- Part 1: Simulate an interruption after the first epoch ---
        class Interruption(Exception):
            pass

        original_should_save = MNISTTrialRunner._should_save_checkpoint
        save_count = 0

        def mock_should_save(self, step):
            nonlocal save_count
            should_save_now = original_should_save(self, step)
            if should_save_now:
                save_count += 1
                if save_count > 1:  # Interrupt before the second save can happen
                    raise Interruption("Simulating crash after first epoch's checkpoint")
            return should_save_now

        monkeypatch.setattr(MNISTTrialRunner, "_should_save_checkpoint", mock_should_save)

        with pytest.raises(Interruption):
            run_experiment_sweep(
                experiment=mnist_config,
                batch_sizes=[64],
                etas=[0.01],
                dataset_loader=mock_mnist_loader,
                directory=str(tmp_path),
                num_epochs=total_epochs,  # Run with the full goal
            )

        # --- Part 2: Verify that the checkpoint from the first epoch exists ---
        cm = CheckpointManager(mnist_config, directory=str(tmp_path))
        resume_file = cm._get_resume_filepath(run_key)
        assert os.path.exists(resume_file)

        # --- Part 3: Restore original behavior and run to completion ---
        monkeypatch.setattr(MNISTTrialRunner, "_should_save_checkpoint", original_should_save)
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

        # --- Part 4: Verify resumption and successful completion ---
        expected_resume_step = steps_per_epoch  # Should resume from step 2 (start of 2nd epoch)
        assert f"Resuming run {run_key} from step {expected_resume_step}" in caplog.text
        assert len(results[run_key]["epoch_test_accuracies"]) == total_epochs
        assert not os.path.exists(resume_file)  # Checkpoint should be cleaned up

    def test_mnist_handles_failed_runs(self, mnist_config, mock_mnist_loader, tmp_path):

        _, failures = run_experiment_sweep(
            experiment=mnist_config,
            batch_sizes=[32],
            etas=[1e20],
            dataset_loader=mock_mnist_loader,
            directory=str(tmp_path),
        )
        assert len(failures) == 1 and RunKey(batch_size=32, eta=1e20) in failures

    def test_optimizer_selection_works(self, mnist_config, mock_mnist_loader, tmp_path):

        from dataclasses import replace

        import jax

        run_key = RunKey(batch_size=64, eta=0.1)
        # The step to check is the last step of the last epoch.
        # num_epochs = 4, steps_per_epoch = 128/64 = 2. Total steps = 8. Last step index = 7.
        num_steps, _ = mnist_config.compute_num_steps(run_key.batch_size, {"image": np.zeros((128, 1))}, 4)
        last_step = num_steps - 1

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
        params_sgd = cm_sgd.load_analysis_snapshot(run_key, step=last_step)

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
        params_adam = cm_adam.load_analysis_snapshot(run_key, step=last_step)

        assert params_sgd is not None and params_adam is not None
        sgd_leaves, _ = jax.tree_util.tree_flatten(params_sgd)
        adam_leaves, _ = jax.tree_util.tree_flatten(params_adam)
        are_different = any(not np.allclose(s, a) for s, a in zip(sgd_leaves, adam_leaves))
        assert are_different, "Final model parameters for SGD and Adam were unexpectedly identical."
