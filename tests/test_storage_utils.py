import os

import pytest

from batch_size_studies.storage_utils import (
    generate_experiment_filename,
    load_experiment,
    save_experiment,
)


@pytest.fixture
def sample_params():
    """A pytest fixture to provide a consistent set of sample parameters."""
    return {"D": 128, "N": 256, "gamma": 1.05, "scale": "SP"}


def test_generate_filename_is_correct_and_consistent(sample_params):
    """Tests that the filename generation is deterministic and formatted correctly."""
    filename = generate_experiment_filename(sample_params)
    expected = "results_D=128_N=256_gamma=1p05_scale=SP.pkl"
    assert filename == expected


def test_generate_filename_with_custom_prefix_and_extension(sample_params):
    """Tests that custom prefixes and extensions are handled correctly."""
    filename = generate_experiment_filename(sample_params, prefix="model_checkpoint", extension="dat")
    expected = "model_checkpoint_D=128_N=256_gamma=1p05_scale=SP.dat"
    assert filename == expected


def test_save_and_load_experiment(sample_params, tmp_path):
    """
    Tests that data can be saved and loaded back correctly.
    Uses pytest's tmp_path fixture to avoid creating files in the project.
    """
    data_to_save = {"losses": {(16, 0.1): [0.5, 0.4]}, "failed_runs": set()}
    directory = tmp_path

    filepath = save_experiment(data_to_save, sample_params, directory, "test_run", "pkl")
    assert os.path.exists(filepath)

    loaded_data = load_experiment(filepath)
    assert loaded_data == data_to_save


def test_load_nonexistent_file_returns_none(tmp_path):
    """Tests that trying to load a file that doesn't exist returns None."""
    non_existent_path = os.path.join(tmp_path, "non_existent_file.pkl")
    assert not os.path.exists(non_existent_path)
    loaded_data = load_experiment(non_existent_path)
    assert loaded_data is None


def test_save_merges_data_correctly(sample_params, tmp_path):
    """Tests that saving without overwrite correctly merges new and existing data."""
    directory = tmp_path

    initial_data = {"losses": {(16, 0.1): [0.5, 0.4]}, "failed_runs": {(32, 1.0)}}
    save_experiment(initial_data, sample_params, directory, "results", "pkl")

    new_data = {"losses": {(16, 0.01): [0.3, 0.2]}, "failed_runs": {(64, 1.0)}}
    filepath = save_experiment(new_data, sample_params, directory, "results", "pkl", overwrite=False)

    merged_data = load_experiment(filepath)

    expected_losses = {(16, 0.1): [0.5, 0.4], (16, 0.01): [0.3, 0.2]}
    expected_failed = {(32, 1.0), (64, 1.0)}

    assert merged_data["losses"] == expected_losses
    assert merged_data["failed_runs"] == expected_failed


def test_save_overwrite_works_correctly(sample_params, tmp_path):
    """Tests that saving with overwrite=True replaces the existing file."""
    directory = tmp_path

    initial_data = {"losses": {(16, 0.1): [0.5, 0.4]}, "failed_runs": set()}
    filepath = save_experiment(initial_data, sample_params, directory, "results", "pkl")

    new_data = {"losses": {(32, 0.01): [0.1, 0.05]}, "failed_runs": set()}
    save_experiment(new_data, sample_params, directory, "results", "pkl", overwrite=True)

    loaded_data = load_experiment(filepath)
    assert loaded_data == new_data
