import argparse
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt

from batch_size_studies.configs import get_small_mup_experiment_configs
from batch_size_studies.data_utils import get_loss_history_from_result
from batch_size_studies.definitions import LossType, RunKey
from batch_size_studies.paths import EXPERIMENTS_DIR


def analyze_results(directory=EXPERIMENTS_DIR):
    """
    Loads results from the experiments and organizes them
    into a nested dictionary structure.

    The structure of the returned dictionary is:
    {
        'experiment_type_1': {
            'loss_label': 'Loss (XENT)',
            'data': {
                gamma_value_1: {
                    N_value_1: {RunKey(bs, eta): [loss_history], ...},
                    ...
                },
            ...
            }
        }
    }

    Args:
        directory (str): The base directory where experiment results are stored.

    Returns:
        dict: A dictionary containing the processed data.
    """
    all_configs = get_small_mup_experiment_configs()

    # Structure: {exp_type: {'data': {...}, 'loss_label': '...'}}
    nested_results = defaultdict(dict)

    print("--- Analyzing Experiment Results ---")
    for name, config in all_configs.items():
        print(f"Loading results for: {name}")

        losses, failed_runs = config.load_results(directory=directory)

        if not losses and not failed_runs:
            print(f"  Warning: No results file found for '{name}'. Skipping.")
            continue

        exp_type = config.experiment_type

        if exp_type not in nested_results:
            loss_name = getattr(config, "loss_type", LossType.MSE).name
            loss_label = f"Loss ({loss_name})"
            nested_results[exp_type] = {
                "data": defaultdict(lambda: defaultdict(dict)),
                "loss_label": loss_label,
            }

        gamma = config.gamma
        N = config.N

        loss_histories = {}
        for run_key, result_obj in losses.items():
            history = get_loss_history_from_result(result_obj)

            if history is None:
                continue

            # For synthetic experiments with num_steps, we can check the length
            if hasattr(config, "num_steps"):
                if len(history) >= config.num_steps:
                    loss_histories[run_key] = history[: config.num_steps]
                else:
                    history_len = len(history)
                    print(
                        f"  Warning: Skipping {run_key} for '{name}' - incomplete loss history "
                        f"(length is {history_len}, expected {config.num_steps})."
                    )
            else:
                # For other types (like MNIST), just accept the history if it exists
                loss_histories[run_key] = history

        if not loss_histories:
            print(f"  No complete loss histories found for '{name}'.")
            continue

        nested_results[exp_type]["data"][gamma][N].update(loss_histories)

    print("\n--- Analysis Complete ---")
    final_results = {}
    for et, data_dict in nested_results.items():
        final_results[et] = {
            "loss_label": data_dict["loss_label"],
            "data": {g: dict(n_dict) for g, n_dict in data_dict["data"].items()},
        }
    return final_results


def plot_losses_for_gamma_and_runkey(results_file, experiment_type, gamma, batch_size, eta):
    """
    Loads the analyzed results and plots the training loss curves for a specific
    gamma and run_key, with one line for each network width (N).
    """
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at '{results_file}'")
        print("Please run the analysis mode first (`--mode analyze`).")
        return

    with open(results_file, "rb") as f:
        results = pickle.load(f)

    if experiment_type not in results:
        print(f"Error: Experiment type '{experiment_type}' not found in results.")
        print(f"Available types: {list(results.keys())}")
        return

    exp_data_bundle = results[experiment_type]
    exp_results = exp_data_bundle["data"]
    loss_label = exp_data_bundle.get("loss_label", "Loss")

    run_key_to_plot = RunKey(batch_size=batch_size, eta=eta)

    if gamma not in exp_results:
        print(f"Error: Gamma value {gamma} not found for experiment type '{experiment_type}'.")
        print(f"Available gammas: {list(exp_results.keys())}")
        return

    gamma_results = exp_results[gamma]

    plt.figure(figsize=(12, 7))

    found_data = False
    sorted_widths = sorted(gamma_results.keys())

    for N in sorted_widths:
        run_data = gamma_results.get(N, {})
        if run_key_to_plot in run_data:
            loss_history = run_data[run_key_to_plot]
            plt.plot(loss_history, label=f"N = {N}")
            found_data = True
        else:
            print(f"Info: RunKey {run_key_to_plot} not found for N={N}.")

    if not found_data:
        print(
            f"\nError: No data for any width w/ RunKey {run_key_to_plot}, exp. type '{experiment_type}' and γ {gamma}."
        )
        plt.close()
        return

    plt.title(f"Training Loss for {experiment_type}, γ={gamma}, B={batch_size}, η={eta}")
    plt.xlabel("Training Step")
    plt.ylabel(loss_label)
    plt.yscale("log")
    plt.legend(title="Width (N)")
    plt.grid(True, which="both", ls="--")

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    exp_type_str = experiment_type.replace("_", "-")
    gamma_str = str(gamma).replace(".", "p")
    bs_str = str(batch_size)
    eta_str = str(eta).replace(".", "p")
    output_filename = f"loss_curves_{exp_type_str}_gamma{gamma_str}_B{bs_str}_eta{eta_str}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path)
    print(f"\nPlot saved to '{output_path}'")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze experiment results or plot specific loss curves.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["analyze", "plot"],
        help="Specify the operation mode.\n"
        "'analyze': Load raw experiment files and save a processed .pkl file.\n"
        "'plot': Load the processed .pkl file and generate a plot for a specific configuration.",
    )
    # Arguments for plotting mode
    parser.add_argument(
        "--experiment_type",
        type=str,
        help="The type of experiment to plot (e.g., 'fixed_time_poly_teacher') (required for --mode=plot).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="The gamma value for the experiment to plot (required for --mode=plot).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The batch size for the specific run to plot (required for --mode=plot).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        help="The learning rate (eta) for the specific run to plot (required for --mode=plot).",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="results/analyzed_small_mup_results.pkl",
        help="Path to the pickled results file for loading or saving.",
    )

    args = parser.parse_args()

    match args.mode:
        case "analyze":
            results = analyze_results()
            if results:
                # Ensure the output directory exists
                output_dir = os.path.dirname(args.results_file)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

                # Save the nested dictionary to a pickle file
                with open(args.results_file, "wb") as f:
                    pickle.dump(results, f)
                print(f"\nAnalyzed results saved to '{args.results_file}'")
            else:
                print("\nNo results were found to analyze.")

        case "plot":
            if not all(
                [args.experiment_type, args.gamma is not None, args.batch_size is not None, args.eta is not None]
            ):
                parser.error("--mode=plot requires --experiment_type, --gamma, --batch_size, and --eta.")

            plot_losses_for_gamma_and_runkey(
                results_file=args.results_file,
                experiment_type=args.experiment_type,
                gamma=args.gamma,
                batch_size=args.batch_size,
                eta=args.eta,
            )
