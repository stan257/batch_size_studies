"""Helpers for listing snapshot steps and computing Hessian spectra."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Iterable

import jax.random as jr
from filelock import FileLock

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import RunKey
from batch_size_studies.storage_utils import CustomUnpickler

from .hessian_evaluator import HessianEvaluator
from .spectral_utils import get_spectral_filepath, load_spectral_data


def list_snapshot_steps(experiment, run_key: RunKey, directory: str) -> list[int]:
    """Return the available snapshot steps for the given run."""
    manager = CheckpointManager(experiment, directory=directory)
    weights_path = manager.weights_filepath
    if not os.path.exists(weights_path):
        logging.warning("Weights file missing for %s; run the experiment first.", weights_path)
        return []
    try:
        with open(weights_path, "rb") as f:
            data = CustomUnpickler(f).load()
        step_map = data.get("weight_snapshots", {}).get(run_key, {})
        return sorted(step_map.keys())
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Failed to inspect snapshot steps: %s", exc)
        return []


def gather_spectra(
    experiment,
    run_key: RunKey,
    steps_to_process: Iterable[int],
    *,
    directory: str,
    spectral_dir: str,
    num_eigenvalues: int,
    num_hessian_samples: int,
    hessian_batch_size: int,
    max_iter: int,
    eig_tol: float,
    trace_samples: int,
    force_recompute: bool = False,
    dry_run: bool = False,
) -> None:
    """Compute Hessian spectra for the requested steps, updating the cache incrementally."""
    spectra_path = get_spectral_filepath(
        experiment,
        directory=directory,
        spectral_dir=spectral_dir,
    )
    spectra_data = load_spectral_data(
        experiment,
        directory=directory,
        spectral_dir=spectral_dir,
    )
    run_dict = spectra_data.setdefault(run_key, {})

    def _persist():
        lock_path = spectra_path + ".lock"
        with FileLock(lock_path):
            # Reload on write to avoid losing updates from concurrent writers.
            if os.path.exists(spectra_path):
                try:
                    with open(spectra_path, "rb") as f:
                        latest_data = CustomUnpickler(f).load()
                except Exception as exc:
                    logging.warning("Failed to reload spectra file; starting fresh: %s", exc)
                    latest_data = {}
            else:
                latest_data = {}

            latest_run_dict = latest_data.setdefault(run_key, {})
            latest_run_dict.update(run_dict)

            tmp_path = spectra_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(latest_data, f)
            os.replace(tmp_path, spectra_path)

    steps_needing_work = []
    for step in steps_to_process:
        stored_vals = run_dict.get(step, {}).get("eigenvalues")
        has_enough = stored_vals is not None and len(stored_vals) >= num_eigenvalues
        if force_recompute or not has_enough:
            steps_needing_work.append(step)

    if dry_run:
        experiment_name = getattr(experiment, "name", getattr(experiment, "experiment_type", "<unknown>"))
        if steps_needing_work:
            logging.info(
                "Dry-run: would compute steps %s for %s %s.",
                steps_needing_work,
                experiment_name,
                run_key,
            )
        else:
            logging.info("Dry-run: all requested steps already cached for %s %s.", experiment_name, run_key)
        return

    for step in steps_to_process:
        stored_vals = run_dict.get(step, {}).get("eigenvalues")
        if stored_vals is not None and len(stored_vals) >= num_eigenvalues and not force_recompute:
            logging.info(
                "Existing entry already has %s eigenvalues; skipping recompute for step %s.",
                len(stored_vals),
                step,
            )
            continue

        logging.info("Evaluating Hessian for step %s", step)
        evaluator = HessianEvaluator(
            experiment=experiment,
            run_key=run_key,
            step=step,
            directory=directory,
            num_hessian_samples=num_hessian_samples,
            hessian_batch_size=hessian_batch_size,
        )
        # Use the evaluator's public API when available so RNG state advances between calls.
        if hasattr(evaluator, "top_eigenvalues"):
            eigenvalues, _ = evaluator.top_eigenvalues(
                top_n=num_eigenvalues,
                max_iter=max_iter,
                tol=eig_tol,
            )
        else:
            eig_key = evaluator.key
            if eig_key is not None:
                eig_key, subkey = jr.split(eig_key)
                evaluator.key = eig_key
                eig_key = subkey
            eigenvalues, _ = evaluator.hessian_computer.eigenvalues(
                evaluator.params,
                eig_key,
                max_iter=max_iter,
                tol=eig_tol,
                top_n=num_eigenvalues,
            )

        if hasattr(evaluator, "trace"):
            trace_value = evaluator.trace(max_iter=trace_samples)
        else:
            trace_key = evaluator.key
            if trace_key is not None:
                trace_key, subkey = jr.split(trace_key)
                evaluator.key = trace_key
                trace_key = subkey
            trace_value, _ = evaluator.hessian_computer.trace(
                evaluator.params,
                trace_key,
                max_iter=trace_samples,
            )

        if stored_vals is not None:
            logging.info("Overwriting existing spectra at step %s (had %s eigenvalues).", step, len(stored_vals))

        run_dict[step] = {
            "eigenvalues": [float(ev) for ev in eigenvalues],
            "trace": float(trace_value),
        }
        _persist()
        logging.info("Saved spectra for step %s -> %s", step, spectra_path)

    _persist()
