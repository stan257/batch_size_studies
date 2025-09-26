"""
Generate HTML Reports from Experiment Results

This script automates the creation of self-contained HTML reports that visualize
the results of hyperparameter sweeps. It loads raw experiment data, generates
various plots (like heatmaps and loss curves) into a single HTML file.
"""

import base64
import os
import types
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt

from batch_size_studies.configs import get_main_experiment_configs, get_main_hyperparameter_grids
from batch_size_studies.data_utils import filter_loss_dicts
from batch_size_studies.experiments import SyntheticExperimentFixedData
from batch_size_studies.paths import EXPERIMENTS_DIR
from batch_size_studies.plotting_utils import plot_loss_curves, plot_loss_heatmap


def _fig_to_base64(fig):
    """
    Converts a matplotlib figure to a base64 encoded string for embedding in HTML.
    This is a best practice for creating self-contained, shareable reports.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


# --- Argument Builders for Plot Registry ---
# These functions decouple the report generation loop from the specific
# argument requirements of each plotting function.


def _build_heatmap_args(loss_dict, plot_title, batch_sizes, etas, base_kwargs):
    plot_args = {
        "loss_dict": loss_dict,
        "title_exp": plot_title,
        "batch_sizes": batch_sizes,
        "etas": etas,
    }
    return plot_args, base_kwargs


def _build_losscurve_args(loss_dict, plot_title, batch_sizes, etas, base_kwargs):
    plot_args = {"loss_dict": loss_dict, "title_exp": plot_title}
    # For automated report generation, we never want to display plots interactively.
    current_kwargs = base_kwargs.copy()
    current_kwargs["display_now"] = False
    return plot_args, current_kwargs


# A Plot Registry separates plot definitions from execution.
# To add a new plot type, simply add a new entry here. The command-line interface will automatically pick it up.
PLOT_REGISTRY = {
    "heatmap_batch": {
        "title": "Batch Space (X-axis: Batch Size, B)",
        "plot_func": plot_loss_heatmap,
        "arg_builder": _build_heatmap_args,
        "plot_kwargs": {"use_ratio_axis": False},
    },
    "heatmap_temp": {
        "title": "Temperature Space (X-axis: Temperature = eta / B)",
        "plot_func": plot_loss_heatmap,
        "arg_builder": _build_heatmap_args,
        "plot_kwargs": {"use_ratio_axis": True},
    },
    # "losscurve_temp_eff_steps": {
    #     "title": "Loss Curves by Temp (X-axis: Effective Steps)",
    #     "plot_func": plot_loss_curves,
    #     "arg_builder": _build_losscurve_args,
    #     "plot_kwargs": {
    #         "group_by": "temp",
    #         "use_eff_steps": True,
    #         "x_scale": "log",
    #     },
    # },
    "losscurve_temp_samples": {
        "title": "Loss Curves by Temp (X-axis: Samples Seen)",
        "plot_func": plot_loss_curves,
        "arg_builder": _build_losscurve_args,
        "plot_kwargs": {
            "group_by": "temp",
            "use_samples_seen": True,
            "x_scale": "log",
        },
        "incompatible_with": [SyntheticExperimentFixedData],
    },
}


def generate_html_report(
    experiments: dict,
    batch_sizes: list[int],
    etas: list[float],
    results_dir: str,
    output_dir: str,
    plots_to_generate: list[str],
):
    """
    Generates a single, self-contained HTML report with embedded heatmap plots.

    Args:
        experiments (dict): A dictionary mapping a descriptive name to an
            experiment configuration object.
        batch_sizes (list[int]): The list of batch sizes for the plot grid.
        etas (list[float]): The list of etas for the plot grid.
        results_dir (str): Root directory where experiment results are stored.
        output_dir (str): Directory to save the HTML report.
        plots_to_generate (list[str]): A list of keys from PLOT_REGISTRY
            specifying which plots to include in the report.
    """
    print(f"--- Generating HTML Report in '{output_dir}' ---")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load all results first to separate data loading from plotting.
    all_results = {}
    for name, config in experiments.items():
        loss_dict, _ = config.load_results(directory=results_dir)
        if loss_dict:
            # Filter the loaded data to match the plotting ranges.
            temp_dict = filter_loss_dicts(loss_dict, filter_by="B", values=batch_sizes)
            filtered_loss_dict = filter_loss_dicts(temp_dict, filter_by="eta", values=etas)
            if filtered_loss_dict:
                all_results[name] = (config, filtered_loss_dict)
            else:
                print(f"  Warning: No data found for '{name}' within the specified batch_sizes and etas. Skipping.")
        else:
            print(f"  Warning: No loss data found for '{name}'. Skipping.")

    if not all_results:
        print("\nNo results found for any experiments. Aborting report generation.")
        return

    # 2. Build the HTML body, grouping by plot type.
    html_body = "<h1>Experiment Heatmap Report</h1>"
    exp_names = ", ".join(f"'{name}'" for name in all_results.keys())
    html_body += f"<p><b>Experiments included:</b> {exp_names}</p><hr>"

    for plot_key in plots_to_generate:
        if plot_key not in PLOT_REGISTRY:
            print(f"  Warning: Plot key '{plot_key}' not found in PLOT_REGISTRY. Skipping.")
            continue
        plot_config = PLOT_REGISTRY[plot_key]
        print(f"\nGenerating plots for: '{plot_config['title']}'...")
        html_body += f"<h2>{plot_config['title']}</h2>"

        for name, (config, loss_dict) in all_results.items():
            # Check for plot compatibility with the experiment type.
            incompatible_types = plot_config.get("incompatible_with", [])
            if any(isinstance(config, t) for t in incompatible_types):
                print(f"  - Skipping '{name}' for plot '{plot_key}': Not applicable for {type(config).__name__}.")
                # Add a placeholder in the HTML report.
                html_body += f"<h3>{name}</h3><p>Plot not applicable for this experiment type.</p>"
                continue

            print(f"  - Plotting '{name}'")
            plot_title = config.plot_title()

            # --- Build arguments using the plot's registered builder ---
            if "arg_builder" not in plot_config:
                print(f"  Error: Plot '{plot_key}' is missing an 'arg_builder' in PLOT_REGISTRY. Skipping.")
                continue

            plot_args, plot_kwargs = plot_config["arg_builder"](
                loss_dict=loss_dict,
                plot_title=plot_title,
                batch_sizes=batch_sizes,
                etas=etas,
                base_kwargs=plot_config["plot_kwargs"],
            )

            # --- Generate plot(s) ---
            # A plotting function can return a single (fig, ax) tuple,
            # or an iterable (list or generator) of them. We normalize to an iterable.
            iterable_or_fig = plot_config["plot_func"](**plot_args, **plot_kwargs)

            if isinstance(iterable_or_fig, (list, types.GeneratorType)):
                fig_ax_iterable = iterable_or_fig
            else:
                # Handle the case of a single (fig, ax) tuple
                fig_ax_iterable = [iterable_or_fig]

            html_body += f"<h3>{name}</h3>"
            plots_generated = False
            for fig, ax in fig_ax_iterable:
                if fig is None:
                    continue
                plots_generated = True
                base64_img = _fig_to_base64(fig)
                html_body += f'<img src="{base64_img}" alt="Plot for {name}" style="width:100%; max-width:800px;">'

            if not plots_generated:
                html_body += "<p>No data to plot for this view.</p>"

        html_body += "<hr>"

    # 4. Assemble and write the final HTML file with a timestamp.
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Experiment Heatmap Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            h1, h2, h3 {{ color: #333; }}
            img {{ border: 1px solid #ddd; border-radius: 4px; padding: 5px; margin-bottom: 2em; }}
            hr {{ border: 1px solid #eee; }}
            p {{ color: #555; }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"report_{timestamp}.html"
    report_path = os.path.join(output_dir, report_filename)

    with open(report_path, "w") as f:
        f.write(html_template)

    print(f"\n--- Report generation complete. View the self-contained report at: '{report_path}' ---")


def main():
    parser = argparse.ArgumentParser(description="Generate a self-contained HTML report of experiment heatmaps.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=EXPERIMENTS_DIR,
        help="Root directory where experiment results are stored. Defaults to the project's 'experiments' directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Directory to save the HTML report. Defaults to 'reports'.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=list(PLOT_REGISTRY.keys()),
        choices=PLOT_REGISTRY.keys(),
        help="Specify which plots to generate. "
        f"Available options: {list(PLOT_REGISTRY.keys())}. Defaults to generating all registered plots.",
    )
    parser.add_argument(
        "-n",
        "--name",
        nargs="*",
        help="Generate reports only for the experiment(s) with these specific names. "
        "If not provided, reports for all defined experiments will be generated.",
    )
    args = parser.parse_args()

    # --- Define the set of experiments and hyperparameter ranges to report on ---
    batch_sizes, etas = get_main_hyperparameter_grids()
    experiments_to_plot = get_main_experiment_configs()

    # Filter experiments by name if provided via the command line.
    if args.name:
        all_names = experiments_to_plot.keys()
        experiments_to_plot = {name: config for name, config in experiments_to_plot.items() if name in args.name}
        if not experiments_to_plot:
            print(f"Error: No experiments found with name(s): {args.name}.")
            print(f"Available experiments are: {list(all_names)}")
            return

    generate_html_report(
        experiments=experiments_to_plot,
        batch_sizes=batch_sizes,
        etas=etas,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        plots_to_generate=args.plots,
    )


if __name__ == "__main__":
    main()
