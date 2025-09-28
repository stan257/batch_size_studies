"""
Run Experimental Gamma-Eta Sweep for MNIST

This script is designed for a specific type of hyperparameter exploration:
a 2D sweep over the richness parameter (gamma) and the learning rate (eta)
for a *fixed batch size*.

It is particularly useful for generating the characteristic phase portrait of
model performance in the γ-η plane from "Optimization Landscape Across Feature
Learning Strength" by Atanasov et al.
"""

import logging

from batch_size_studies.definitions import LossType, Parameterization
from batch_size_studies.experimental_mnist_training import run_mnist_gamma_eta_sweep
from batch_size_studies.experiments import MNIST1MSampledExperiment
from batch_size_studies.paths import EXPERIMENTS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Run a gamma-eta hyperparameter sweep for MNIST experiments with a fixed batch size."
    )

    parser.add_argument("--N", type=int, default=128, help="Hidden layer width of the MLP.")
    parser.add_argument("--L", type=int, default=3, help="Number of layers in the MLP.")
    parser.add_argument(
        "--parameterization",
        type=str,
        choices=[p.value for p in Parameterization],
        default=Parameterization.MUP.value,
        help="Model parameterization (SP or muP).",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=[lt.value for lt in LossType],
        default=LossType.MSE.value,
        help="Loss function to use (MSE or XENT).",
    )
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs for each trial.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=50_000,
        help="Number of training samples to use (from MNIST-1M). If 0, use the full dataset.",
    )

    parser.add_argument("--batch-size", type=int, required=True, help="Fixed batch size for all trials.")
    parser.add_argument("--gamma-range", type=int, default=5, help="Log10 range for the gamma sweep.")
    parser.add_argument("--gamma-res", type=int, default=2, help="Resolution (points per decade) for the gamma sweep.")
    parser.add_argument("--eta-range", type=int, default=13, help="Log10 range for the eta sweep.")
    parser.add_argument("--eta-res", type=int, default=2, help="Resolution for the eta sweep.")
    parser.add_argument(
        "--logspace-eta-range",
        type=int,
        default=3,
        help="How many decades to search below the first stable learning rate.",
    )

    parser.add_argument("--init-key", type=int, default=0, help="Base random key for parameter initialization.")
    parser.add_argument(
        "--save-subfolder",
        type=str,
        default="sanity_checks",
        help="Subfolder within the experiment directory to save results.",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=EXPERIMENTS_DIR,
        help="Root directory for saving all experiment data.",
    )
    parser.add_argument("--no-save", action="store_true", help="If set, do not save any results or checkpoints.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    base_experiment = MNIST1MSampledExperiment(
        N=args.N,
        L=args.L,
        parameterization=Parameterization(args.parameterization),
        loss_type=LossType(args.loss_type),
        num_epochs=args.num_epochs,
        max_train_samples=args.max_train_samples if args.max_train_samples > 0 else None,
    )

    logging.info("--- Starting Experiment Sweep ---")
    logging.info(f"Base experiment: {base_experiment}")
    logging.info(
        f"Sweep parameters: Batch Size={args.batch_size}, Gamma Range={args.gamma_range}, Eta Range={args.eta_range}"
    )

    results, failures = run_mnist_gamma_eta_sweep(
        base_experiment=base_experiment,
        batch_size=args.batch_size,
        gamma_range=args.gamma_range,
        gamma_res=args.gamma_res,
        eta_range=args.eta_range,
        eta_res=args.eta_res,
        logspace_eta_range=args.logspace_eta_range,
        init_key=args.init_key,
        directory=args.directory,
        save_subfolder=args.save_subfolder,
        no_save=args.no_save,
    )

    logging.info("--- Sweep Finished ---")
    successful_runs = sum(len(v) for v in results.values())
    failed_runs = sum(len(v) for v in failures.values())
    logging.info(f"Successful runs: {successful_runs}")
    logging.info(f"Failed runs: {failed_runs}")


if __name__ == "__main__":
    main()
