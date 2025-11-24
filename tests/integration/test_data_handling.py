from dataclasses import replace

import jax
import jax.random as jr
import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import LossType, OptimizerType, RunKey
from batch_size_studies.experiments import (
    SyntheticExperimentLinearTeacher,
    SyntheticExperimentNoisyLinearTeacher,
)
from batch_size_studies.hessian_evaluator import HessianEvaluator
from batch_size_studies.runner import run_experiment_sweep


@pytest.fixture
def linear_teacher_config_integration():
    """A config for integration testing data handling."""
    return SyntheticExperimentLinearTeacher(
        D=1,
        P=1000,  # Dataset size that is not perfectly divisible by all test batch sizes
        alpha=1.0,
        beta=1.0,
        seed=42,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
        num_epochs=1,  # Default to 1, will be overridden in tests
    )


class TestDataHandlingIntegration:
    def test_data_subset_and_uniqueness_across_epochs(self, linear_teacher_config_integration, monkeypatch):
        """
        This integration test verifies two critical data handling properties:
        1. Single-Epoch Run: No data points are repeated within a single epoch.
        2. Multi-Epoch Run: The exact same subset of data points is used for
           every epoch, just in a different order.
        """
        batch_sizes = [4, 8, 16, 32]
        etas = [0.01]  # A single eta is sufficient
        P = linear_teacher_config_integration.P

        # This dictionary will store the indices seen for each run
        # Structure: {batch_size: {epoch: [indices]}}
        seen_data_log = {}

        # --- 1. Monkeypatch the data generation and training loop ---

        # Replace the training loop to just collect batch indices
        def mock_run_training_loop(self, params, opt_state, results, start_step, data_iterator):
            # This mock replaces the actual training.
            # It iterates through the data generator for each epoch and logs the indices.
            batch_size = self.run_key.batch_size
            if batch_size not in seen_data_log:
                seen_data_log[batch_size] = {}

            # The iterator yields all batches for all epochs in a single sequence.
            # We consume the passed iterator once and then partition the results by epoch for the test.
            all_batches = list(data_iterator)
            num_total_batches = len(all_batches)

            for epoch in range(self.num_epochs):
                start_idx = epoch * self.steps_per_epoch
                end_idx = (epoch + 1) * self.steps_per_epoch

                # Check if there are any batches left for this epoch
                if start_idx >= num_total_batches:
                    break

                # Extract the batches for the current epoch
                epoch_data_batches = [x_batch for x_batch, _ in all_batches[start_idx:end_idx]]

                if not epoch_data_batches:
                    break
                epoch_data = np.vstack(epoch_data_batches)
                seen_data_log[batch_size][epoch] = epoch_data

            # Return a minimal valid result to satisfy the runner
            results["loss_history"] = [0.1] * self.num_steps
            return params, opt_state, results

        monkeypatch.setattr(
            "batch_size_studies.trainer.SyntheticFixedDataTrialRunner._run_training_loop", mock_run_training_loop
        )

        # --- 2. Run the single-epoch experiment ---
        single_epoch_config = replace(linear_teacher_config_integration, num_epochs=1)
        run_experiment_sweep(experiment=single_epoch_config, batch_sizes=batch_sizes, etas=etas, no_save=True)

        # Store the results from the single-epoch run for later comparison
        single_epoch_data = {bs: seen_data_log[bs][0] for bs in batch_sizes}

        # --- 3. Verify the single-epoch run ---
        for bs in batch_sizes:
            data = single_epoch_data[bs]
            num_usable_samples = (P // bs) * bs

            # Assert correct number of samples were used
            assert data.shape[0] == num_usable_samples, (
                f"For B={bs}, expected {num_usable_samples} samples, but got {data.shape[0]}"
            )

            # Assert that all samples seen in the epoch are unique
            # For floating point arrays, we check uniqueness of rows.
            unique_data = np.unique(data, axis=0)
            assert unique_data.shape[0] == data.shape[0], (
                f"For B={bs}, duplicate data points were found within the single epoch."
            )

        # --- 4. Run the multi-epoch experiment ---
        seen_data_log.clear()
        multi_epoch_config = replace(linear_teacher_config_integration, num_epochs=3)
        run_experiment_sweep(experiment=multi_epoch_config, batch_sizes=batch_sizes, etas=etas, no_save=True)

        # --- 5. Verify the multi-epoch run ---
        for bs in batch_sizes:
            # The set of indices used in the single-epoch run
            base_subset_sorted = np.sort(single_epoch_data[bs], axis=0)

            # The indices seen in each epoch of the multi-epoch run
            epoch0_data = seen_data_log[bs][0]
            epoch1_data = seen_data_log[bs][1]
            epoch2_data = seen_data_log[bs][2]

            # Assert that the set of indices is the same across all epochs
            np.testing.assert_allclose(
                np.sort(epoch0_data, axis=0),
                base_subset_sorted,
                err_msg=f"For B={bs}, the data subset in epoch 0 of the multi-epoch run differs from the single-epoch run.",
            )
            np.testing.assert_allclose(
                np.sort(epoch1_data, axis=0),
                base_subset_sorted,
                err_msg=f"For B={bs}, the data subset in epoch 1 differs from epoch 0.",
            )
            np.testing.assert_allclose(
                np.sort(epoch2_data, axis=0),
                base_subset_sorted,
                err_msg=f"For B={bs}, the data subset in epoch 2 differs from epoch 0.",
            )

            # Assert that the order is different between epochs (shuffling works)
            assert not np.array_equal(epoch0_data, epoch1_data), (
                f"For B={bs}, the data order in epoch 0 and 1 should be different."
            )
            assert not np.array_equal(epoch1_data, epoch2_data), (
                f"For B={bs}, the data subset in epoch 0 of the multi-epoch run differs from the single-epoch run."
            )

    def test_noisy_linear_teacher_matches_clean_for_zero_noise(self):
        base = SyntheticExperimentLinearTeacher(
            D=32,
            P=2048,
            alpha=1.0,
            beta=1.0,
            optimizer=OptimizerType.SGD,
            loss_type=LossType.MSE,
            num_epochs=1,
            seed=123,
        )
        noisy = SyntheticExperimentNoisyLinearTeacher(
            D=32,
            P=2048,
            alpha=1.0,
            beta=1.0,
            optimizer=OptimizerType.SGD,
            loss_type=LossType.MSE,
            num_epochs=1,
            seed=123,
            rho=0.0,
        )
        key = jr.PRNGKey(321)
        X_base, y_base = base.generate_data(key)
        X_noisy, y_noisy = noisy.generate_data(key)
        np.testing.assert_allclose(X_noisy, X_base)
        np.testing.assert_allclose(y_noisy, y_base)

    def test_noisy_linear_teacher_reproducibility(self, tmp_path):
        config = SyntheticExperimentNoisyLinearTeacher(
            D=16,
            P=2048,
            alpha=1.0,
            beta=1.0,
            optimizer=OptimizerType.SGD,
            loss_type=LossType.MSE,
            num_epochs=2,
            seed=99,
            rho=0.4,
        )
        batch_sizes = [32]
        etas = [0.05]

        def run_once(directory):
            run_experiment_sweep(
                experiment=config,
                batch_sizes=batch_sizes,
                etas=etas,
                directory=directory,
                init_key=123,
                no_save=False,
            )
            results, fails = config.load_results(directory=directory)
            manager = CheckpointManager(config, directory=directory)
            run_key = RunKey(batch_size=batch_sizes[0], eta=etas[0])
            weights = manager.load_full_weight_history(run_key)
            data_key = jr.PRNGKey(1234)
            data = config.generate_data(data_key)
            return results, fails, weights, data

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        res1, fails1, weights1, data1 = run_once(str(dir1))
        res2, fails2, weights2, data2 = run_once(str(dir2))

        assert fails1 == fails2
        assert res1.keys() == res2.keys()
        for key in res1:
            np.testing.assert_allclose(res1[key]["loss_history"], res2[key]["loss_history"])
        assert weights1.keys() == weights2.keys()
        for step in weights1:
            flat1 = jax.tree_util.tree_leaves(weights1[step])
            flat2 = jax.tree_util.tree_leaves(weights2[step])
            for arr1, arr2 in zip(flat1, flat2):
                np.testing.assert_allclose(np.asarray(arr1), np.asarray(arr2))
        (X1, y1), (X2, y2) = data1, data2
        np.testing.assert_allclose(X1, X2)
        np.testing.assert_allclose(y1, y2)

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
    def test_noisy_linear_teacher_signal_to_noise_matches_empirical(self, rho):
        config = SyntheticExperimentNoisyLinearTeacher(
            D=64,
            P=5000,
            alpha=1.0,
            beta=1.0,
            optimizer=OptimizerType.SGD,
            loss_type=LossType.MSE,
            num_epochs=1,
            seed=2024,
            rho=rho,
        )

        key = jr.PRNGKey(777)
        X, y = config.generate_data(key)
        w = config.generate_teacher_weights()
        raw_signal = X @ w
        signal_component = np.sqrt(1.0 - rho) * raw_signal
        noise_component = y - signal_component
        signal_var = float(np.var(signal_component))
        noise_var = float(np.var(noise_component))
        empirical_ratio = np.inf if noise_var == 0 else signal_var / noise_var
        expected_ratio = config.signal_to_noise()
        if np.isinf(expected_ratio):
            assert np.isinf(empirical_ratio)
        else:
            np.testing.assert_allclose(empirical_ratio, expected_ratio, rtol=5e-3)

    def test_hessian_evaluator_reads_checkpoint_and_matches_eigenvalue(self, tmp_path):
        config = SyntheticExperimentLinearTeacher(
            D=16,
            P=512,
            alpha=1.0,
            beta=1.0,
            optimizer=OptimizerType.SGD,
            loss_type=LossType.MSE,
            num_epochs=1,
            seed=7,
        )
        batch_sizes = [32]
        etas = [0.1]
        directory = tmp_path / "experiments"

        run_experiment_sweep(
            experiment=config,
            batch_sizes=batch_sizes,
            etas=etas,
            directory=str(directory),
            init_key=0,
        )

        manager = CheckpointManager(config, directory=str(directory))
        run_key = RunKey(batch_size=batch_sizes[0], eta=etas[0])
        history = manager.load_full_weight_history(run_key)
        assert history, "Expected at least one weight snapshot for the linear teacher sweep."
        target_step = min(history.keys())

        evaluator = HessianEvaluator(
            experiment=config,
            run_key=run_key,
            step=target_step,
            directory=str(directory),
            num_hessian_samples=config.P,
            hessian_batch_size=32,
        )

        eigenvalues, _ = evaluator.top_eigenvalues(top_n=1, max_iter=200, tol=1e-4)
        np.testing.assert_allclose(np.array(eigenvalues[0]), 1.0, rtol=7e-2)

        expected_trace = np.sum(np.arange(1, config.D + 1, dtype=np.float64) ** (-config.alpha))
        trace_val = evaluator.trace(max_iter=128)
        np.testing.assert_allclose(np.array(trace_val), expected_trace, rtol=1e-1)
