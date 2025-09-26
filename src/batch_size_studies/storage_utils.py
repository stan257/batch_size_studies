import os
import pickle


class CustomUnpickler(pickle.Unpickler):
    """
    A custom unpickler that handles module path changes for backward
    compatibility with older pickle files.
    """

    def find_class(self, module, name):
        # Remap 'definitions' to 'batch_size_studies.definitions'
        if module == "definitions":
            module = "batch_size_studies.definitions"
        return super().find_class(module, name)


def generate_experiment_filename(params, prefix="results", extension="pkl"):
    """Generates a standardized, deterministic filename from a params dict."""
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
    """Loads experiment data from a pickle file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            return CustomUnpickler(f).load()
    except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
        print(f"Warning: Could not load or unpickle file: {filepath}. Error: {e}")
        return None


def save_experiment(data_to_save, params, directory, prefix, extension, overwrite=False):
    """
    Saves a data dictionary, safely merging with existing data by default.
    """
    filepath = os.path.join(directory, generate_experiment_filename(params, prefix, extension))

    # If not overwriting, merge with existing data
    if not overwrite and os.path.exists(filepath):
        existing_data = load_experiment(filepath)
        if existing_data:
            # Deep merge logic for the specific structure {'losses': ..., 'failed_runs': ...}
            merged_losses = existing_data.get("losses", {})
            merged_losses.update(data_to_save.get("losses", {}))

            merged_failed = existing_data.get("failed_runs", set())
            merged_failed.update(data_to_save.get("failed_runs", set()))

            data_to_save = {"losses": merged_losses, "failed_runs": merged_failed}

    os.makedirs(directory, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data_to_save, f)
    return filepath
