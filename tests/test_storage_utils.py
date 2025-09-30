import os
from unittest.mock import patch

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
    directory = str(tmp_path)
    filename = generate_experiment_filename(sample_params, "test_run", "pkl")
    filepath = os.path.join(directory, filename)

    save_experiment(data_to_save, filepath)
    assert os.path.exists(filepath)

    loaded_data = load_experiment(filepath)
    assert loaded_data == data_to_save


def test_load_nonexistent_file_returns_none(tmp_path):
    """Tests that trying to load a file that doesn't exist returns None."""
    non_existent_path = str(tmp_path / "non_existent_file.pkl")
    assert not os.path.exists(non_existent_path)
    loaded_data = load_experiment(non_existent_path)
    assert loaded_data is None


def test_save_overwrites_existing_file(sample_params, tmp_path):
    """Tests that saving again to the same path overwrites the file."""
    directory = str(tmp_path)
    filename = generate_experiment_filename(sample_params, "results", "pkl")
    filepath = os.path.join(directory, filename)

    initial_data = {"losses": {(16, 0.1): [0.5, 0.4]}, "failed_runs": {(32, 1.0)}}
    save_experiment(initial_data, filepath)

    new_data = {"losses": {(32, 0.01): [0.1, 0.05]}, "failed_runs": set()}
    save_experiment(new_data, filepath)

    loaded_data = load_experiment(filepath)
    assert loaded_data == new_data


@patch("batch_size_studies.storage_utils.os.replace")
@patch("batch_size_studies.storage_utils.pickle.dump")
def test_atomic_write_sequence(mock_pickle_dump, mock_os_replace, sample_params, tmp_path):
    """Tests that save_experiment uses the atomic write pattern (write to .tmp, then rename)."""
    data_to_save = {"test": "data"}
    directory = str(tmp_path)
    filename = generate_experiment_filename(sample_params, "results", "pkl")
    filepath = os.path.join(directory, filename)
    temp_filepath = filepath + ".tmp"

    save_experiment(data_to_save, filepath)

    mock_pickle_dump.assert_called_once()
    mock_os_replace.assert_called_once_with(temp_filepath, filepath)


@patch("batch_size_studies.storage_utils.pickle.dump", side_effect=IOError("Disk full"))
def test_atomic_write_cleans_up_on_failure(mock_pickle_dump, sample_params, tmp_path):
    """Tests that the temporary file is removed if an error occurs during writing."""
    data_to_save = {"test": "data"}
    directory = str(tmp_path)
    filename = generate_experiment_filename(sample_params, "results", "pkl")
    filepath = os.path.join(directory, filename)
    temp_filepath = filepath + ".tmp"

    with pytest.raises(IOError):
        save_experiment(data_to_save, filepath)

    assert not os.path.exists(temp_filepath)
