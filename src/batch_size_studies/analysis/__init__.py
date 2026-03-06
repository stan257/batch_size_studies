"""Canonical analysis namespace for results processing and plotting."""

from .plotting import plot_all_loss_curves, plot_heatmap_with_theory_curve, plot_loss_curves, plot_loss_heatmap
from .results import (
    adjust_run_keys_in_dict,
    extract_loss_histories,
    filter_experiments,
    filter_loss_dict_by_cutoff,
    filter_loss_dict_by_loss_threshold,
    filter_loss_dicts,
    get_first_divergence_eta,
    moving_average,
)

__all__ = [
    "plot_loss_heatmap",
    "plot_heatmap_with_theory_curve",
    "plot_loss_curves",
    "plot_all_loss_curves",
    "moving_average",
    "filter_loss_dicts",
    "filter_loss_dict_by_cutoff",
    "filter_loss_dict_by_loss_threshold",
    "extract_loss_histories",
    "filter_experiments",
    "get_first_divergence_eta",
    "adjust_run_keys_in_dict",
]
