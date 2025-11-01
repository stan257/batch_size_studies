import itertools
from dataclasses import replace

import numpy as np
import pytest

from batch_size_studies.definitions import LossType, OptimizerType
from batch_size_studies.experiments import SyntheticExperimentLinearTeacher
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
        def mock_run_training_loop(self, params, opt_state, results, start_step):
            # This mock replaces the actual training.
            # It iterates through the data generator for each epoch and logs the indices.
            batch_size = self.run_key.batch_size
            if batch_size not in seen_data_log:
                seen_data_log[batch_size] = {}

            # The new generator yields across all epochs. We need to manually chunk it.
            full_generator = self._create_data_generator(results, start_step)

            for epoch in range(self.num_epochs):
                epoch_data_batches = [x_batch for x_batch, _ in itertools.islice(full_generator, self.steps_per_epoch)]
                epoch_data = np.vstack(epoch_data_batches)
                seen_data_log[batch_size][epoch] = epoch_data

            # Return a minimal valid result to satisfy the runner
            results["loss_history"] = [0.1] * (self.num_epochs * (self.num_train // batch_size))
            return results

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
