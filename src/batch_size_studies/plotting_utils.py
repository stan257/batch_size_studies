from collections import defaultdict
from typing import Any, Callable, Literal

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns  # noqa: F401 needed for cmap='rocket'

from .data_utils import get_loss_history_from_result
from .definitions import RunKey


def _default_final_metric_extractor(result: Any) -> float | None:
    """Default metric_extractor for heatmaps, gets the last loss value."""
    history = get_loss_history_from_result(result)
    if history is not None and len(history) > 0:
        return history[-1]
    return None


def _prepare_grouped_curves(
    loss_dict: dict[RunKey, Any],
    group_by: Literal["B", "eta", "temp"],
    use_eff_steps: bool,
    use_samples_seen: bool,
) -> dict[Any, list[dict]]:
    """
    Pre-processes a loss dictionary into structured, plottable curve data,
    grouped by a specified parameter.

    This function separates data preparation from plotting logic. It handles:
    - Extracting loss histories.
    - Calculating x-axis values (steps, effective steps, or samples seen).
    - Grouping curves by batch size, eta, or temperature.

    Returns:
        A dictionary where keys are group values (e.g., a batch size) and
        values are lists of curve data dictionaries. Each curve dictionary
        contains 'x_values', 'y_values', and run parameters 'B' and 'eta'.
    """
    grouped_curves = defaultdict(list)

    for run_key, result_obj in loss_dict.items():
        loss_values = get_loss_history_from_result(result_obj)
        if loss_values is None or len(loss_values) == 0:
            continue

        b, eta = run_key.batch_size, run_key.eta
        group_value = None

        # Determine group value
        match group_by:
            case "B":
                group_value = b
            case "eta":
                group_value = eta
            case "temp":
                if b > 0:
                    group_value = round(eta / b, 8)

        if group_value is None:
            continue

        # --- Data Preparation ---
        y_values = loss_values
        if not len(y_values):
            continue

        steps = np.arange(len(loss_values)) + 1
        if use_eff_steps:
            x_values = steps * eta
        elif use_samples_seen:
            x_values = steps * b
        else:
            x_values = steps

        curve_data = {"x_values": x_values, "y_values": y_values, "B": b, "eta": eta}
        grouped_curves[group_value].append(curve_data)

    return grouped_curves


def moving_average(data: np.ndarray, window_size: int):
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def plot_loss_heatmap(
    loss_dict: dict[RunKey, Any],
    batch_sizes: list[int],
    etas: list[float],
    title_exp: str,
    metric_extractor: Callable[[Any], float | None] = _default_final_metric_extractor,
    use_ratio_axis: bool = False,
    clim: tuple[float, float] | None = None,
    cmap: str = "rocket",
    ax=None,
):
    """
    Generates a formatted 2D heatmap of loss values.

    Can plot with batch_size (B) on the x-axis or the temperature (eta / B).

    Args:
        loss_dict (dict): Maps a RunKey to a result object (e.g., a list of
            losses or a dict containing metrics).
        batch_sizes (list or np.array): The list of batch sizes.
        etas (list or np.array): The list of learning rates (eta).
        title_exp (str): The title for the experiment associated with the plot.
        metric_extractor (Callable, optional): A function that takes the value from
            the results dictionary and returns the scalar metric to plot.
            Defaults to extracting the last item from a list.
        use_ratio_axis (bool, optional): If True, x-axis is the temperature (eta / B).
            Defaults to False, which uses batch_size (B) on the x-axis.
        clim (tuple[float, float] | None, optional): The color limits (min, max)
            for the log10-scaled loss. If None, the limits are automatically
            inferred from the data. Defaults to None.
        cmap (str, optional): The colormap to use. Defaults to 'rocket'.
        ax (matplotlib.axes.Axes, optional): An existing axes object to plot on.
                                             If None, a new figure and axes are created.
                                             Defaults to None.

    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects (fig, ax).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # --- 1. Prepare data grid and index lookups for efficiency ---
    Z = np.full((len(etas), len(batch_sizes)), np.nan, dtype=float)
    b_to_i = {b: i for i, b in enumerate(batch_sizes)}
    eta_to_j = {e: j for j, e in enumerate(etas)}

    # --- 2. Populate the grid by iterating through data once ---
    for run_key, result_obj in loss_dict.items():
        b, eta = run_key.batch_size, run_key.eta
        if b in b_to_i and eta in eta_to_j:
            i = b_to_i[b]
            j = eta_to_j[eta]
            metric_val = metric_extractor(result_obj)
            if metric_val is not None and metric_val > 0:
                Z[j, i] = np.log10(metric_val)

    # --- 3. Plotting logic (mostly unchanged) ---
    X, Y = np.meshgrid(batch_sizes, etas)
    # Axes are in base 2
    Y_coords = np.log2(Y)

    if use_ratio_axis:
        # X-axis is the temperature eta / B
        X_coords = np.log2(Y / X)
        xlabel = "$\\eta / B$"
        ax.set_xlim(np.nanmin(X_coords), np.nanmax(X_coords))
    else:
        # X-axis is Batch Size B
        X_coords = np.log2(X)
        xlabel = "$B$"
        ax.set_xlim(min(np.log2(batch_sizes)), max(np.log2(batch_sizes)))

    pcm = ax.pcolormesh(X_coords, Y_coords, Z, cmap=cmap, shading="auto")

    # Determine color limits automatically if not provided
    if clim is None:
        if np.any(np.isfinite(Z)):
            vmin, vmax = np.nanmin(Z), np.nanmax(Z)
            pcm.set_clim(vmin, vmax)
    else:
        pcm.set_clim(clim)
    ax.set_ylim(min(np.log2(etas)), max(np.log2(etas)))

    def log2_tick_formatter(val, pos=None):
        return f"$2^{{{round(val)}}}$"

    def log10_tick_formatter(val, pos=None):
        return f"$10^{{{round(val)}}}$"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log2_tick_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log2_tick_formatter))

    # Set integer ticks for the colorbar to avoid repeated labels from rounding.
    vmin, vmax = pcm.get_clim()
    ticks = np.arange(np.ceil(vmin), np.floor(vmax) + 1, dtype=int)

    cbar = fig.colorbar(pcm, ax=ax, ticks=ticks, format=mticker.FuncFormatter(log10_tick_formatter))
    cbar.set_label("Loss")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("$\\eta$")
    ax.set_title("Online Loss in " + title_exp)

    return fig, ax


def plot_loss_curves(
    loss_dict: dict[RunKey, Any],
    title_exp: str = "",
    group_by: str = "B",
    group_values: list | Any | None = None,
    plot_on_single_ax: bool = False,
    use_eff_steps: bool = False,
    use_samples_seen: bool = False,
    x_scale: str = "linear",
    y_scale: str = "log",
    cmap: str = "rocket",
    ax=None,
    display_now: bool = True,
):
    """
    Plots loss curves over training steps, grouped by a specified parameter.

    Can be used in two modes:
    1. Multi-plot generator (default, `plot_on_single_ax=False`):
       Generates a separate plot for each group.
       - If `group_values` is None, discovers all unique groups.
       - If `group_values` is a list, iterates through them, creating one plot per value.
       - This mode is a generator, yielding (fig, ax) tuples unless `display_now` is True.

    2. Single-plot (`plot_on_single_ax=True`):
       Plots multiple groups on a single axis for comparison.
       - `group_values` must be a list of values to plot.
       - Returns a single (fig, ax) tuple.

    Args:
        loss_dict (dict): Maps RunKey to a result object from which a loss
            history can be extracted.
        title_exp (str): The title for the experiment.
        group_by (str): Parameter to group curves by ('B', 'eta', or 'temp').
        group_values (list|Any|None): Specific value(s) for the group to plot.
            If None, all unique values are discovered and plotted.
        plot_on_single_ax (bool): If True, plots all groups on a single axis.
        use_eff_steps (bool): If True, x-axis is effective steps (steps * eta).
        use_samples_seen (bool): If True, x-axis is samples seen (steps * B).
        x_scale (str): Matplotlib scale for the x-axis (e.g., 'linear', 'log').
        y_scale (str): Matplotlib scale for the y-axis (e.g., 'linear', 'log').
        cmap (str): The colormap for coloring curves within a group.
        display_now (bool): If True, display plots immediately in an interactive
                            session (like a notebook) instead of yielding them.
        ax (matplotlib.axes.Axes, optional): An existing axes object to plot on
            for single-axis mode.

    Yields:
        tuple: In multi-plot mode, yields (fig, ax) for each generated plot.

    Returns:
        tuple: In single-plot mode, returns a single (fig, ax) tuple.
    """
    assert not (use_eff_steps and use_samples_seen), "Only one of use_eff_steps or use_samples_seen can be True."

    # --- 1. Prepare and group data using the new helper ---
    grouped_curves = _prepare_grouped_curves(
        loss_dict,
        group_by=group_by,
        use_eff_steps=use_eff_steps,
        use_samples_seen=use_samples_seen,
    )

    # --- Helper to plot a pre-grouped set of curves ---
    def _plot_group(ax, value_to_fix, curves_in_group, linestyle="-"):
        if not curves_in_group:
            return False  # Indicate no plots were made

        # Sort curves for consistent coloring
        sort_key = "B" if group_by == "eta" else "eta"
        curves_in_group.sort(key=lambda x: x[sort_key])

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(curves_in_group)))

        # Determine x-axis label once
        if use_eff_steps:
            xlabel = "Effective Steps (steps * η)"
        elif use_samples_seen:
            xlabel = "Samples Seen (steps * B)"
        else:
            xlabel = "Training Steps"

        # Plot each curve
        for i, curve in enumerate(curves_in_group):
            b, eta = curve["B"], curve["eta"]
            label = ""
            # Generate label based on context
            match group_by:
                case "B":
                    eta_exp = round(np.log2(eta)) if eta > 0 else -np.inf
                    label = f"B={b}, $\\eta=2^{{{eta_exp}}}$" if plot_on_single_ax else f"$\\eta = 2^{{{eta_exp}}}$"
                case "eta":
                    b_exp = round(np.log2(b)) if b > 0 else -np.inf
                    label = f"$\\eta={eta:.2g}, B=2^{{{b_exp}}}$" if plot_on_single_ax else f"$B = 2^{{{b_exp}}}$"
                case "temp":
                    b_exp = round(np.log2(b)) if b > 0 else -np.inf
                    eta_exp = round(np.log2(eta)) if eta > 0 else -np.inf
                    label = (
                        f"η/B≈{value_to_fix:.2g}, B=$2^{{{b_exp}}}$"
                        if plot_on_single_ax
                        else f"$B=2^{{{b_exp}}}, \\eta=2^{{{eta_exp}}}$"
                    )

            ax.plot(
                curve["x_values"],
                curve["y_values"],
                label=label,
                color=colors[i],
                linestyle=linestyle,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss")
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        return True  # Indicate plots were made

    # --- Main function logic ---

    # 2. Determine the list of group values to iterate over
    if group_values is None:
        # Discover unique values from the pre-grouped data
        values_to_plot = sorted(list(grouped_curves.keys()))
    else:
        # Coerce group_values to a flat list of scalars to handle various inputs gracefully.
        if isinstance(group_values, np.ndarray):
            values_to_plot = group_values.flatten().tolist()
        elif not isinstance(group_values, list):
            values_to_plot = [group_values]  # Handle single scalar value
        else:
            values_to_plot = group_values

    if not values_to_plot:
        print("Warning: No data in loss_dict to plot for the given groups.")
        if not plot_on_single_ax:
            return  # Empty generator
        else:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
            return fig, ax

    # 3. Execute plotting based on the mode
    if plot_on_single_ax:
        # --- Single-axis mode ---
        if not isinstance(values_to_plot, list) or len(values_to_plot) < 1:
            raise ValueError("`plot_on_single_ax=True` requires `group_values` to be a list of one or more values.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        linestyles = ["-", "--", ":", "-."]
        for i, value in enumerate(values_to_plot):
            ls = linestyles[i % len(linestyles)]
            curves_for_group = grouped_curves.get(value, [])
            _plot_group(ax, value, curves_for_group, linestyle=ls)

        # Create title for multi-group plot
        group_name_map = {"B": "Batch Size B", "eta": "η", "temp": "Temp η/B"}
        group_name = group_name_map.get(group_by, group_by)

        def format_value(v, group):
            match group:
                case "temp":
                    return f"{float(v):.2e}"
                case "eta":
                    return f"{float(v):.3g}"
                case _:
                    return str(v)

        formatted_values = ", ".join([format_value(v, group_by) for v in values_to_plot])
        group_title = f"Loss Curves for {group_name} values: {formatted_values}"
        full_title = f"{group_title}\n{title_exp}" if title_exp else group_title
        ax.set_title(full_title)

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        return fig, ax

    else:
        # --- Multi-plot generator mode ---
        def plot_generator():
            if display_now:
                try:
                    from IPython.display import display
                except ImportError:
                    print(
                        "Warning: `display_now=True` requires IPython, which couldn't be imported. Fallback plt.show()."
                    )
                    display = plt.show

            for value in values_to_plot:
                fig, ax = plt.subplots()

                # Create title for single-group plot
                match group_by:
                    case "B":
                        group_title = f"Loss Curves for Batch Size B = {value}"
                    case "eta":
                        group_title = f"Loss Curves for η = {float(value):.3g}"
                    case "temp":
                        group_title = f"Loss Curves for Temp η/B ≈ {float(value):.3g}"
                    case _:
                        group_title = f"Loss Curves for {group_by} = {value}"

                full_title = f"{group_title}\n{title_exp}" if title_exp else group_title
                ax.set_title(full_title)

                curves_for_group = grouped_curves.get(value, [])
                if _plot_group(ax, value, curves_for_group):
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                    fig.tight_layout(rect=[0, 0, 0.85, 1])
                else:
                    ax.text(0.5, 0.5, "No data found", ha="center", va="center")

                if display_now:
                    display(fig)
                    plt.close(fig)
                else:
                    yield (fig, ax)

        # Create the generator object.
        generator = plot_generator()

        if display_now:
            # If display_now is True, consume the generator to display the plots.
            # The generator itself handles displaying and doesn't yield anything.
            for _ in generator:
                pass
        else:
            # Otherwise, return the generator for the user to iterate over.
            return generator


def plot_all_loss_curves(
    loss_dict: dict[RunKey, Any],
    title_exp: str = "",
    plot_group_by: str = "temp",
    use_eff_steps: bool = False,
    use_samples_seen: bool = False,
    x_scale: str = "linear",
    y_scale: str = "log",
    ax=None,
):
    """
    Plots all loss curves on a single axis with a structured grid legend.

    Each curve is given a unique color. Curves are grouped by a parameter
    ('B', 'eta', or 'temp'), and all curves in a group share the same colormap.
    The legend is a 2D grid corresponding to (Batch Size, Eta) pairs.

    Args:
        loss_dict (dict): Maps a RunKey to a result object. The result object
            can be a list of losses or a dictionary containing a
            'loss_history' key.
        title_exp (str): The title for the experiment.
        plot_group_by (str): Parameter to group curves by for colormap
            ('B', 'eta', or 'temp').
        use_eff_steps (bool): If True, x-axis is effective steps (steps * eta).
        use_samples_seen (bool): If True, x-axis is samples seen (steps * B).
        x_scale (str): Matplotlib scale for the x-axis.
        y_scale (str): Matplotlib scale for the y-axis.
        ax (matplotlib.axes.Axes, optional): An existing axes to plot on.

    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects (fig, ax).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    assert not (use_eff_steps and use_samples_seen), "Only one of use_eff_steps or use_samples_seen can be True."

    # 1. Prepare all curve data using the new helper.
    # We group by 'B' arbitrarily; we just need to process all curves.
    # The result is a dict of lists, so we flatten it.
    grouped_curves_data = _prepare_grouped_curves(
        loss_dict,
        group_by="B",  # Can be any valid group_by, we just need to process all curves
        use_eff_steps=use_eff_steps,
        use_samples_seen=use_samples_seen,
    )
    curves_to_plot = [curve for group in grouped_curves_data.values() for curve in group]

    if not curves_to_plot:
        print("Warning: No valid loss histories found in loss_dict.")
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        return fig, ax

    # 2. Group curves by the specified parameter
    grouped_curves = defaultdict(list)
    for curve in curves_to_plot:
        b, eta = curve["B"], curve["eta"]
        group_val = None
        match plot_group_by:
            case "B":
                group_val = b
            case "eta":
                group_val = eta
            case "temp":
                if b > 0:
                    group_val = round(eta / b, 8)
        if group_val is not None:
            grouped_curves[group_val].append(curve)

    # 3. Setup colormaps and legend grid.
    # Generate a set of distinct base colors for the groups.
    num_groups = len(grouped_curves)
    # Using a qualitative palette like "deep" from seaborn provides good separation.
    base_colors = sns.color_palette("deep", num_groups)

    all_bs = sorted(list({c["B"] for c in curves_to_plot}))
    all_etas = sorted(list({c["eta"] for c in curves_to_plot}))

    # --- Dynamically orient the legend grid for a better aspect ratio ---
    # The parameter with more unique values will become the rows.
    transpose_legend = len(all_etas) > len(all_bs)

    if transpose_legend:
        # Taller legend: rows are eta (asc), cols are B (asc)
        row_items, col_items = all_etas, all_bs
        item_to_row = {item: i for i, item in enumerate(row_items)}  # eta_to_row
        item_to_col = {item: i for i, item in enumerate(col_items)}  # b_to_col
        legend_title = "(log₂(η), log₂(B))"
    else:
        # Wider/Square legend: rows are B (asc), cols are eta (asc)
        row_items, col_items = all_bs, all_etas
        item_to_row = {item: i for i, item in enumerate(row_items)}  # b_to_row
        item_to_col = {item: i for i, item in enumerate(col_items)}  # eta_to_col
        legend_title = "(log₂(B), log₂(η))"

    sorted_groups = sorted(grouped_curves.keys())

    # 4. Prepare legend placeholders and figure sizing
    num_rows, num_cols = len(row_items), len(col_items)

    # --- Dynamic figure sizing to ensure fixed plot area ---
    # Define desired fixed size for the plot axes in inches
    AXES_WIDTH_IN = 8
    AXES_HEIGHT_IN = 6

    # Estimate legend width based on number of columns (heuristic)
    legend_width_in = num_cols * 1.2

    # Define padding around the plot and legend
    left_margin_in = 1.2  # For y-axis label and ticks
    right_margin_in = 0.5  # Space after legend
    top_margin_in = 1.0  # For title
    bottom_margin_in = 1.0  # For x-axis label and ticks

    # Calculate total figure dimensions
    fig_width_in = left_margin_in + AXES_WIDTH_IN + legend_width_in + right_margin_in
    fig_height_in = bottom_margin_in + AXES_HEIGHT_IN + top_margin_in

    fig.set_size_inches(fig_width_in, fig_height_in)

    # Adjust subplot to fix axes size and make space for legend
    left = left_margin_in / fig_width_in
    right = (left_margin_in + AXES_WIDTH_IN) / fig_width_in
    bottom = bottom_margin_in / fig_height_in
    top = (bottom_margin_in + AXES_HEIGHT_IN) / fig_height_in
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    empty_handle = mlines.Line2D([], [], color="none", marker="None", linestyle="None", label="")
    legend_handles = [[empty_handle] * num_cols for _ in range(num_rows)]
    legend_labels = [[""] * num_cols for _ in range(num_rows)]

    # 5. Main plotting loop
    if use_eff_steps:
        xlabel = "Effective Steps (steps * η)"
    elif use_samples_seen:
        xlabel = "Samples Seen (steps * B)"
    else:
        xlabel = "Training Steps"

    for i, group_val in enumerate(sorted_groups):
        curves_in_group = grouped_curves[group_val]

        # Sort curves within the group for consistent coloring
        sort_key = "B" if plot_group_by == "eta" else "eta"
        curves_in_group.sort(key=lambda c: c[sort_key])

        # Create a sequential colormap from the group's base color.
        base_color = base_colors[i]
        cmap_for_group = sns.light_palette(base_color, as_cmap=True)

        # Use a range that avoids the lightest colors for better visibility.
        num_curves_in_group = len(curves_in_group)
        colors_for_group = cmap_for_group(np.linspace(0.4, 1.0, num_curves_in_group))
        for j, curve in enumerate(curves_in_group):
            color = colors_for_group[j]
            b, eta = curve["B"], curve["eta"]

            # Plot the curve
            (line,) = ax.plot(curve["x_values"], curve["y_values"], color=color)

            # Populate legend information
            if transpose_legend:
                row, col = item_to_row[eta], item_to_col[b]
            else:
                row, col = item_to_row[b], item_to_col[eta]

            legend_handles[row][col] = line
            b_exp = round(np.log2(b)) if b > 0 else -np.inf
            eta_exp = round(np.log2(eta)) if eta > 0 else -np.inf
            legend_labels[row][col] = f"({eta_exp}, {b_exp})" if transpose_legend else f"({b_exp}, {eta_exp})"

    # 6. Create and place the custom grid legend
    # Matplotlib's legend fills column-major. To achieve a visual row-major
    # layout, we must flatten our handle and label matrices in column-major order.
    if not legend_handles:
        flat_handles, flat_labels = [], []
    else:
        transposed_handles = list(zip(*legend_handles))
        flat_handles = [item for sublist in transposed_handles for item in sublist]

        transposed_labels = list(zip(*legend_labels))
        flat_labels = [item for sublist in transposed_labels for item in sublist]

    ax.legend(
        flat_handles,
        flat_labels,
        ncol=num_cols,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        title=legend_title,
        fontsize="small",
        handlelength=1.5,
    )

    # 7. Finalize plot
    ax.set_title(f"Loss Curves\n{title_exp}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss")
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)

    # Add explanation for coloring.
    # This text is placed relative to the axes, below the x-axis label,
    # and horizontally aligned with the legend.
    group_name_map = {"B": "batch size", "eta": "learning rate", "temp": "temperature (η/B)"}
    group_name = group_name_map.get(plot_group_by, plot_group_by)
    footer_text = f"*Each color family (e.g., shades of blue) corresponds to runs with same {group_name}."
    ax.text(
        1.05,
        -0.12,  # Position it below the x-axis label
        footer_text,
        transform=ax.transAxes,
        fontsize="small",
        va="top",
        ha="left",
        style="italic",
    )

    return fig, ax
