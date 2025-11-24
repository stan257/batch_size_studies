import matplotlib.pyplot as plt
import pytest

from batch_size_studies.definitions import RunKey
from batch_size_studies.plotting_utils import (
    _final_validation_error_extractor,
    plot_heatmap_with_theory_curve,
    plot_loss_curves,
)


def test_final_validation_error_extractor_prefers_accuracy():
    result = {
        "final_test_accuracy": 0.8,
        "epoch_test_accuracies": [0.5, 0.6],
        "final_eval_loss": 0.2,
        "loss_history": [1.0, 0.9],
    }
    assert pytest.approx(_final_validation_error_extractor(result)) == 0.2  # 1 - 0.8

    del result["final_test_accuracy"]
    assert pytest.approx(_final_validation_error_extractor(result)) == 0.4  # 1 - 0.6

    del result["epoch_test_accuracies"]
    assert pytest.approx(_final_validation_error_extractor(result)) == 0.2


def _simple_loss_dict():
    return {
        RunKey(16, 0.1): {"loss_history": [1.0, 0.5]},
        RunKey(32, 0.05): {"loss_history": [0.8, 0.4]},
    }


def test_plot_heatmap_with_theory_curve_returns_fig():
    loss_dict = _simple_loss_dict()
    fig, ax = plot_heatmap_with_theory_curve(
        loss_dict=loss_dict,
        batch_sizes=[16, 32],
        etas=[0.1, 0.05],
        title_exp="demo",
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_plot_heatmap_logs_warning_for_non_positive_metric(caplog):
    loss_dict = {
        RunKey(16, 0.1): {"loss_history": [1.0]},
        RunKey(32, 0.05): {"loss_history": [0.0]},  # non positive
    }
    with caplog.at_level("WARNING"):
        fig, ax = plot_heatmap_with_theory_curve(
            loss_dict=loss_dict,
            batch_sizes=[16, 32],
            etas=[0.1, 0.05],
            title_exp="warn",
        )
    warning_messages = [record.getMessage() for record in caplog.records]
    assert any("non-positive" in msg for msg in warning_messages)
    plt.close(fig)


def test_plot_heatmap_logs_when_lower_bound_fails(caplog):
    loss_dict = _simple_loss_dict()

    def flaky_lower_bound(batch_size: int) -> float:
        raise ValueError("boom")

    with caplog.at_level("WARNING"):
        fig, ax = plot_heatmap_with_theory_curve(
            loss_dict=loss_dict,
            batch_sizes=[16, 32],
            etas=[0.1, 0.05],
            title_exp="bounds",
            lower_bound=flaky_lower_bound,
        )
    assert any("lower_bound callable failed" in record.message for record in caplog.records)
    plt.close(fig)


def test_plot_loss_curves_single_axis():
    loss_dict = _simple_loss_dict()
    fig, ax = plot_loss_curves(
        loss_dict=loss_dict,
        title_exp="curves",
        group_by="B",
        group_values=[16],
        plot_on_single_ax=True,
        display_now=False,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)
