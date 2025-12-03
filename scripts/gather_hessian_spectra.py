#!/usr/bin/env python3
"""Compute and cache Hessian spectra for specific runs and checkpoints."""

import argparse
import logging
import os
import pickle

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.configs import get_main_experiment_configs
from batch_size_studies.definitions import RunKey
from batch_size_studies.hessian_evaluator import HessianEvaluator
from batch_size_studies.paths import EXPERIMENTS_DIR, SPECTRAL_DATA_DIR
from batch_size_studies.spectral_utils import get_spectral_filepath, load_spectral_data
from batch_size_studies.storage_utils import CustomUnpickler


def _load_experiment(name: str):
    configs = get_main_experiment_configs()
    config = configs.get(name)
    if config is None:
        raise KeyError(f"Experiment '{name}' not found. Did you run scripts/run_experiments.py?")
    return config


def _available_snapshot_steps(manager: CheckpointManager, run_key: RunKey) -> list[int]:
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


def _persist_spectra(filepath: str, data: dict) -> None:
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_path, filepath)


def compute_spectrum(args) -> None:
    experiment = _load_experiment(args.experiment)
    run_key = RunKey(batch_size=args.batch_size, eta=args.eta)
    manager = CheckpointManager(experiment, directory=args.experiments_dir)

    available_steps = _available_snapshot_steps(manager, run_key)
    if not available_steps:
        logging.error(
            "No snapshots found for %s (B=%s, eta=%s). Enable save-interstitial-snapshots and rerun the experiment.",
            args.experiment,
            args.batch_size,
            args.eta,
        )
        return

    logging.info("Snapshot steps for %s %s: %s", args.experiment, run_key, available_steps)
    if args.list_only:
        return

    steps_to_process = args.steps if args.steps else available_steps
    missing = set(steps_to_process) - set(available_steps)
    if missing:
        raise ValueError(f"Requested steps {sorted(missing)} are not available. Pick from {available_steps}.")

    spectra_path = get_spectral_filepath(
        experiment,
        directory=args.experiments_dir,
        spectral_dir=SPECTRAL_DATA_DIR,
    )
    spectra_data = load_spectral_data(
        experiment,
        directory=args.experiments_dir,
        spectral_dir=SPECTRAL_DATA_DIR,
    )
    run_dict = spectra_data.setdefault(run_key, {})
    steps_needing_work = []
    for step in steps_to_process:
        stored_vals = run_dict.get(step, {}).get("eigenvalues")
        has_enough = stored_vals is not None and len(stored_vals) >= args.num_eigenvalues
        if args.force_recompute or not has_enough:
            steps_needing_work.append(step)

    if args.dry_run:
        if steps_needing_work:
            logging.info(
                "Dry-run: would compute steps %s for %s %s.",
                steps_needing_work,
                args.experiment,
                run_key,
            )
        else:
            logging.info(
                "Dry-run: all requested steps already cached for %s %s.",
                args.experiment,
                run_key,
            )
        return

    for step in steps_to_process:
        stored_vals = run_dict.get(step, {}).get("eigenvalues")
        if stored_vals is not None and len(stored_vals) >= args.num_eigenvalues and not args.force_recompute:
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
            directory=args.experiments_dir,
            num_hessian_samples=args.num_hessian_samples,
            hessian_batch_size=args.hessian_batch_size,
        )
        eigenvalues, _ = evaluator.hessian_computer.eigenvalues(
            evaluator.params,
            evaluator.key,
            max_iter=args.max_iter,
            tol=args.eig_tol,
            top_n=args.num_eigenvalues,
        )
        trace_value, _ = evaluator.hessian_computer.trace(
            evaluator.params,
            evaluator.key,
            max_iter=args.trace_samples,
        )
        new_vals = [float(ev) for ev in eigenvalues]
        if stored_vals is not None:
            logging.info("Overwriting existing spectra at step %s (had %s eigenvalues).", step, len(stored_vals))

        run_dict[step] = {
            "eigenvalues": new_vals,
            "trace": float(trace_value),
        }
        _persist_spectra(spectra_path, spectra_data)
        logging.info("Saved spectra for step %s -> %s", step, spectra_path)

    # Final write to ensure the file reflects any in-memory updates
    _persist_spectra(spectra_path, spectra_data)


def parse_args():
    parser = argparse.ArgumentParser(description="Gather Hessian spectra for a specific run and checkpoints.")
    parser.add_argument("--experiment", required=True, help="Experiment name as registered in configs.")
    parser.add_argument("--batch-size", type=int, required=True, dest="batch_size", help="Batch size of the run.")
    parser.add_argument("--eta", type=float, required=True, help="Learning rate eta of the run.")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        help="Specific snapshot steps to evaluate. Defaults to all available steps if omitted.",
    )
    parser.add_argument("--list-only", action="store_true", help="Only list available steps without computing spectra.")
    parser.add_argument("--experiments-dir", default=EXPERIMENTS_DIR, help="Directory housing experiment results.")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute spectra even if cached entries already have >= num-eigenvalues.",
    )
    parser.add_argument("--num-eigenvalues", type=int, default=100, help="Number of eigenvalues to compute.")
    parser.add_argument("--num-hessian-samples", type=int, default=1024, help="Samples used for Hessian estimation.")
    parser.add_argument("--hessian-batch-size", type=int, default=128, help="Batch size for Hessian-vector products.")
    parser.add_argument("--max-iter", type=int, default=100, help="Iterations for power iteration eigen solver.")
    parser.add_argument("--eig-tol", type=float, default=1e-3, help="Tolerance for eigen solver convergence.")
    parser.add_argument("--trace-samples", type=int, default=100, help="Iterations for Hutchinson trace estimator.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which steps would be computed without running Hessian evaluations.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    compute_spectrum(args)


if __name__ == "__main__":
    main()
