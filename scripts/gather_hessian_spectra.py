#!/usr/bin/env python3
"""Compute and cache Hessian spectra for specific runs and checkpoints."""

import argparse
import logging

from batch_size_studies.configs import get_main_experiment_configs
from batch_size_studies.definitions import RunKey
from batch_size_studies.paths import EXPERIMENTS_DIR, SPECTRAL_DATA_DIR
from batch_size_studies.spectral import gather_spectra, list_snapshot_steps


def _load_experiment(name: str):
    configs = get_main_experiment_configs()
    config = configs.get(name)
    if config is None:
        raise KeyError(f"Experiment '{name}' not found. Did you run scripts/run_experiments.py?")
    return config


def compute_spectrum(args) -> None:
    experiment = _load_experiment(args.experiment)
    run_key = RunKey(batch_size=args.batch_size, eta=args.eta)

    available_steps = list_snapshot_steps(experiment, run_key, args.experiments_dir)
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

    gather_spectra(
        experiment,
        run_key,
        steps_to_process,
        directory=args.experiments_dir,
        spectral_dir=SPECTRAL_DATA_DIR,
        num_eigenvalues=args.num_eigenvalues,
        num_hessian_samples=args.num_hessian_samples,
        hessian_batch_size=args.hessian_batch_size,
        max_iter=args.max_iter,
        eig_tol=args.eig_tol,
        trace_samples=args.trace_samples,
        force_recompute=args.force_recompute,
        dry_run=args.dry_run,
    )


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
