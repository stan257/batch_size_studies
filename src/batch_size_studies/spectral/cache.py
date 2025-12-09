"""Spectral cache helpers for Hessian pipeline."""

from __future__ import annotations

import os
import pickle

from filelock import FileLock

from ..storage_utils import CustomUnpickler
from .spectral_utils import get_spectral_filepath, load_spectral_data


class SpectralCache:
    """Manages loading and persisting Hessian spectra entries."""

    def __init__(self, experiment, *, directory: str, spectral_dir: str | None = None):
        self.experiment = experiment
        self.directory = directory
        self.spectral_dir = spectral_dir
        self.filepath = get_spectral_filepath(experiment, directory=directory, spectral_dir=spectral_dir)
        self._lock_path = self.filepath + ".lock"
        self._data: dict | None = None

    def _load_data(self) -> dict:
        if self._data is None:
            self._data = load_spectral_data(self.experiment, directory=self.directory, spectral_dir=self.spectral_dir)
        return self._data

    def get_run_dict(self, run_key):
        data = self._load_data()
        return data.setdefault(run_key, {})

    def store_step(self, run_key, step: int, eigenvalues: list[float], trace_value: float):
        entry = {"eigenvalues": [float(ev) for ev in eigenvalues], "trace": float(trace_value)}
        self.get_run_dict(run_key)[step] = entry

        with FileLock(self._lock_path):
            latest = {}
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, "rb") as f:
                        latest = CustomUnpickler(f).load() or {}
                except Exception:
                    latest = {}

            latest.setdefault(run_key, {})[step] = entry

            tmp_path = self.filepath + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(latest, f)
            os.replace(tmp_path, self.filepath)
