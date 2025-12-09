"""Helpers for listing snapshot steps and computing Hessian spectra."""

from __future__ import annotations

import logging
import os
from typing import Iterable

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import RunKey
from batch_size_studies.storage_utils import CustomUnpickler

from .cache import SpectralCache
from .hessian_evaluator import HessianEvaluator


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
    cache = SpectralCache(
        experiment,
        directory=directory,
        spectral_dir=spectral_dir,
    )
    spectra_path = cache.filepath
    run_dict = cache.get_run_dict(run_key)

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
        eigenvalues, _ = evaluator.top_eigenvalues(
            top_n=num_eigenvalues,
            max_iter=max_iter,
            tol=eig_tol,
        )
        trace_value = evaluator.trace(max_iter=trace_samples)

        if stored_vals is not None:
            logging.info("Overwriting existing spectra at step %s (had %s eigenvalues).", step, len(stored_vals))

        cache.store_step(run_key, step, eigenvalues, trace_value)
        logging.info("Saved spectra for step %s -> %s", step, spectra_path)
