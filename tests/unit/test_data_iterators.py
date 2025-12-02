from unittest.mock import Mock

import jax.random as jr
import numpy as np
import pytest

from batch_size_studies.data_iterators import EpochBasedDataIterator, OnlineDataIterator


class TestEpochBasedDataIterator:
    @pytest.fixture
    def sample_data(self):
        # Create a simple, deterministic dataset of unique integers
        return np.arange(105).reshape(-1, 1)

    def test_data_subsetting(self, sample_data):
        iterator = EpochBasedDataIterator(train_ds=(sample_data, sample_data), batch_size=32, num_epochs=1, init_key=42)

        # Expected number of samples: (105 // 32) * 32 = 3 * 32 = 96
        num_usable_samples = (105 // 32) * 32
        assert len(iterator.subset_indices) == num_usable_samples

        # Collect all data yielded by the iterator
        all_yielded_data = np.vstack([x_batch for x_batch, _ in iterator])

        assert all_yielded_data.shape[0] == num_usable_samples
        # Verify that all yielded samples are unique
        assert np.unique(all_yielded_data).shape[0] == num_usable_samples

    def test_epoch_shuffling_and_consistency(self, sample_data):
        iterator = EpochBasedDataIterator(train_ds=(sample_data, sample_data), batch_size=32, num_epochs=3, init_key=42)

        all_batches = list(iterator)
        steps_per_epoch = 105 // 32

        # Extract data for each epoch
        epoch1_data = np.vstack([x for x, _ in all_batches[0:steps_per_epoch]])
        epoch2_data = np.vstack([x for x, _ in all_batches[steps_per_epoch : 2 * steps_per_epoch]])

        # 1. Assert that the set of data is the same
        np.testing.assert_array_equal(np.sort(epoch1_data, axis=0), np.sort(epoch2_data, axis=0))

        # 2. Assert that the order is different (shuffling worked)
        assert not np.array_equal(epoch1_data, epoch2_data)

    def test_resumption_logic(self, sample_data):
        batch_size = 32
        steps_per_epoch = 105 // 32  # 3
        start_step = 4  # This is the 2nd step of the 2nd epoch (epoch 1, step_in_epoch 1)

        # First, get the full, uninterrupted sequence of data
        full_iterator = EpochBasedDataIterator(
            train_ds=(sample_data, sample_data), batch_size=batch_size, num_epochs=2, init_key=42
        )
        full_sequence = [x_batch for x_batch, _ in full_iterator]

        # Now, create an iterator that should resume
        resuming_iterator = EpochBasedDataIterator(
            train_ds=(sample_data, sample_data),
            batch_size=batch_size,
            num_epochs=2,
            init_key=42,
            start_step=start_step,
        )
        resumed_sequence = [x_batch for x_batch, _ in resuming_iterator]

        # The first batch from the resumed sequence should be the 5th batch (index 4) from the full sequence
        np.testing.assert_array_equal(resumed_sequence[0], full_sequence[start_step])

        # The total number of batches should be correct
        total_steps = 2 * steps_per_epoch  # 6
        assert len(resumed_sequence) == total_steps - start_step  # 6 - 4 = 2

    def test_resumption_uses_saved_epoch_seed(self, sample_data):
        batch_size = 16
        steps_per_epoch = 105 // 16
        epoch = 1
        step_in_epoch = 2
        start_step = epoch * steps_per_epoch + step_in_epoch
        custom_seed = 777

        iterator = EpochBasedDataIterator(
            train_ds=(sample_data, sample_data),
            batch_size=batch_size,
            num_epochs=3,
            init_key=42,
            start_step=start_step,
            resume_state={"epoch": epoch, "step_in_epoch": step_in_epoch, "epoch_seed": custom_seed},
        )

        first_batch, _ = next(iter(iterator))
        subset = iterator.subset_indices
        perms = jr.permutation(jr.PRNGKey(custom_seed), subset).reshape((steps_per_epoch, batch_size))
        expected_indices = np.array(perms[step_in_epoch])
        np.testing.assert_array_equal(first_batch, sample_data[expected_indices])

    def test_dict_dataset_batches_are_flattened(self):
        num_samples = 64
        batch_size = 16
        images = np.arange(num_samples * 6).reshape(num_samples, 2, 3)
        labels = np.arange(num_samples)
        train_ds = {"image": images, "label": labels}

        iterator = EpochBasedDataIterator(
            train_ds=train_ds,
            batch_size=batch_size,
            num_epochs=1,
            init_key=0,
        )

        batches = list(iterator)
        assert len(batches) == num_samples // batch_size

        first_images, first_labels = batches[0]
        assert first_images.shape == (batch_size, 6)
        np.testing.assert_array_equal(first_labels.shape, (batch_size,))

    def test_handles_batch_size_larger_than_dataset(self, sample_data):
        batch_size = sample_data.shape[0] + 10
        iterator = EpochBasedDataIterator(
            train_ds=(sample_data, sample_data),
            batch_size=batch_size,
            num_epochs=1,
            init_key=0,
        )

        assert list(iterator) == []


class TestOnlineDataIterator:
    @pytest.fixture
    def mock_experiment(self):
        """A mock experiment that we can spy on."""
        mock_exp = Mock()
        mock_exp.P = 100  # Dataset size per key

        # The generate_data method will return deterministic, unique data based on the key
        def generate_data(key):
            # This mock is designed to return different data based on the key.
            # The correct way to check for key equality is to compare their data arrays.
            key_data = jr.key_data(key)

            if np.array_equal(key_data, jr.key_data(jr.PRNGKey(0))):
                start_val = 0
            elif np.array_equal(key_data, jr.key_data(jr.PRNGKey(1))):
                start_val = 1000
            elif np.array_equal(key_data, jr.key_data(jr.PRNGKey(2))):
                start_val = 2000
            else:
                start_val = -1  # Default for any other keys
            data = np.arange(start_val, start_val + mock_exp.P).reshape(-1, 1)
            return data, data

        mock_exp.generate_data = Mock(side_effect=generate_data)
        return mock_exp

    def test_on_the_fly_generation_and_seed_increment(self, mock_experiment):
        batch_size = 30
        iterable = OnlineDataIterator(
            experiment=mock_experiment, batch_size=batch_size, start_step=0, initial_batch_key_seed=0
        )
        iterator = iter(iterable)

        # Iterate 7 times to cross multiple data generation boundaries
        for _ in range(7):
            next(iterator)  # Call next on the iterator object

        # It should have called generate_data for keys 0, 1, and 2
        assert mock_experiment.generate_data.call_count == 3
        # Compare the underlying data of the JAX keys
        call_keys_data = [jr.key_data(call.args[0]) for call in mock_experiment.generate_data.call_args_list]
        np.testing.assert_array_equal(call_keys_data[0], jr.key_data(jr.PRNGKey(0)))
        np.testing.assert_array_equal(call_keys_data[1], jr.key_data(jr.PRNGKey(1)))
        np.testing.assert_array_equal(call_keys_data[2], jr.key_data(jr.PRNGKey(2)))

    def test_resumption_logic(self, mock_experiment):
        batch_size = 30
        start_step = 5  # This is the 3rd step (index 2) using data from key=1

        iterable = OnlineDataIterator(
            experiment=mock_experiment, batch_size=batch_size, start_step=start_step, initial_batch_key_seed=0
        )
        iterator = iter(iterable)

        # The first call to next() should trigger data generation for the correct key
        first_batch, _ = next(iterator)  # Call next on the iterator object

        # 1. Verify it started with the correct key (fast-forwarded)
        mock_experiment.generate_data.assert_called_once()
        # Compare the underlying data of the JAX keys
        first_call_key_data = jr.key_data(mock_experiment.generate_data.call_args.args[0])
        np.testing.assert_array_equal(first_call_key_data, jr.key_data(jr.PRNGKey(1)))

        # 2. Verify it yielded the correct slice of data.
        # step_for_curr_data = start_step % (mock_experiment.P // batch_size) = 5 % 3 = 2.
        # The slice starts at index 2 * 30 = 60.
        # So the first value should be 1000 + 60 = 1060.
        assert first_batch[0, 0] == 1060

    def test_handles_batch_size_larger_than_p(self, mock_experiment):
        mock_experiment.P = 50
        batch_size = 100  # B > P

        iterable = OnlineDataIterator(
            experiment=mock_experiment, batch_size=batch_size, start_step=0, initial_batch_key_seed=0
        )

        all_batches = list(iterable)
        assert len(all_batches) == 0
        mock_experiment.generate_data.assert_not_called()

    def test_handles_zero_batch_size(self, mock_experiment):
        iterable = OnlineDataIterator(experiment=mock_experiment, batch_size=0, start_step=0, initial_batch_key_seed=0)

        assert list(iterable) == []
        mock_experiment.generate_data.assert_not_called()
