import logging
import os
import pickle

# =============================================================================
# Pickle helpers and filename utilities
# =============================================================================


_LEGACY_EXPERIMENT_MODULE = "batch_size_studies.experiments"
_EXPERIMENT_CLASS_REMAP = {
    "ExperimentBase": "batch_size_studies.experiment_types.base",
    "MLPStudentExperiment": "batch_size_studies.experiment_types.base",
    "LinearStudentExperiment": "batch_size_studies.experiment_types.base",
    "SyntheticExperiment": "batch_size_studies.experiment_types.synthetic",
    "SyntheticExperimentFixedTime": "batch_size_studies.experiment_types.synthetic",
    "SyntheticExperimentFixedData": "batch_size_studies.experiment_types.synthetic",
    "SyntheticExperimentMLPTeacher": "batch_size_studies.experiment_types.synthetic",
    "SyntheticExperimentLinearTeacher": "batch_size_studies.experiment_types.synthetic",
    "SyntheticExperimentNoisyLinearTeacher": "batch_size_studies.experiment_types.synthetic",
    "MNISTExperiment": "batch_size_studies.experiment_types.mnist",
    "MNIST1MExperiment": "batch_size_studies.experiment_types.mnist",
    "MNIST1MSampledExperiment": "batch_size_studies.experiment_types.mnist",
}


class CustomUnpickler(pickle.Unpickler):
    """
    A custom unpickler that handles module path changes for backward
    compatibility with older pickle files.
    """

    def find_class(self, module, name):
        # Remap 'definitions' to 'batch_size_studies.definitions'
        if module == "definitions":
            module = "batch_size_studies.definitions"
        if module == _LEGACY_EXPERIMENT_MODULE and name in _EXPERIMENT_CLASS_REMAP:
            module = _EXPERIMENT_CLASS_REMAP[name]
        return super().find_class(module, name)


def generate_experiment_filename(params, prefix="results", extension="pkl"):
    filename_parts = [prefix]
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            value_str = str(value).replace(".", "p")
        else:
            value_str = str(value)
        filename_parts.append(f"{key}={value_str}")
    base_name = "_".join(filename_parts)
    return f"{base_name}.{extension}"


def load_experiment(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            return CustomUnpickler(f).load()
    except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
        logging.warning(f"Could not load or unpickle file: {filepath}. Error: {e}")
        return None


def save_experiment(data_to_save, filepath: str):
    """
    Saves a data dictionary to a given filepath using an atomic write.

    This function writes to a temporary file first, then renames it to the
    final destination. This prevents file corruption if the process is
    interrupted during the write operation.
    """
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    temp_filepath = filepath + ".tmp"
    try:
        with open(temp_filepath, "wb") as f:
            pickle.dump(data_to_save, f)
        # os.replace is atomic on most systems
        os.replace(temp_filepath, filepath)
    except Exception as e:
        logging.error(f"Failed to save data to {filepath}: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise
    return filepath
