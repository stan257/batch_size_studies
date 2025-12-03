import os
from pathlib import Path

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.paths import SPECTRAL_DATA_DIR
from batch_size_studies.storage_utils import CustomUnpickler


def get_spectral_filepath(experiment, directory=None, spectral_dir: str | None = None) -> str:
    """
    Resolves the canonical filepath for Hessian spectra associated with an experiment.
    """
    manager = CheckpointManager(experiment, directory=directory)
    weights_name = os.path.basename(manager.weights_filepath)
    spectra_name = weights_name.replace("_weights.pkl", "_spectra.pkl")
    spectral_root = Path(spectral_dir or SPECTRAL_DATA_DIR) / experiment.experiment_type
    spectral_root.mkdir(parents=True, exist_ok=True)
    return str(spectral_root / spectra_name)


def load_spectral_data(experiment, directory=None, spectral_dir: str | None = None) -> dict:
    """
    Loads cached Hessian spectra for the given experiment.
    """
    filepath = get_spectral_filepath(experiment, directory=directory, spectral_dir=spectral_dir)
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "rb") as f:
            return CustomUnpickler(f).load()
    except Exception:  # pylint: disable=broad-except
        return {}
