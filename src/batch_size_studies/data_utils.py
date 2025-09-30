from functools import wraps
from typing import Any, Callable, Type, TypeVar

import numpy as np

from .definitions import LossType, OptimizerType, RunKey
from .experiments import ExperimentBase

D = TypeVar("D", bound=dict)


def apply_to_list_of_dicts(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(data: D | list[D], *args, **kwargs) -> D | list[D]:
        if isinstance(data, list):
            return [func(item, *args, **kwargs) for item in data]
        return func(data, *args, **kwargs)

    return wrapper


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def get_loss_history_from_result(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("loss_history")
    if isinstance(result, (list, np.ndarray)):
        return result
    # A simple check for other array-like objects (like JAX arrays)
    if hasattr(result, "shape") and hasattr(result, "dtype"):
        return result
    return None


def _get_run_key_value(rk: RunKey, by: str) -> float | int | None:
    if by == "B":
        return rk.batch_size
    if by == "eta":
        return rk.eta
    if by == "temp":
        temp = rk.temp
        return round(temp, 8) if temp is not None else None
    return None


@apply_to_list_of_dicts
def filter_loss_dicts(loss_dict: dict, filter_by: str, values: list, keep: bool = True) -> dict:
    """Filters a dictionary by keeping or removing specified values for a given parameter."""
    if filter_by not in ("B", "eta", "temp"):
        raise ValueError("filter_by must be one of 'B', 'eta', or 'temp'")

    value_set = {round(v, 8) if filter_by == "temp" else v for v in values}

    return {
        rk: val
        for rk, val in loss_dict.items()
        if (v := _get_run_key_value(rk, filter_by)) is not None and (v in value_set) == keep
    }


@apply_to_list_of_dicts
def uniform_smooth_loss_dicts(loss_dict: dict[RunKey, Any], smoother: Callable[[Any], Any]) -> dict[RunKey, Any]:
    return {
        run_key: smoother(loss_history)
        for run_key, result_obj in loss_dict.items()
        if (loss_history := get_loss_history_from_result(result_obj)) is not None
    }


@apply_to_list_of_dicts
def sample_aware_smooth_loss_dicts(
    loss_dict: dict[RunKey, Any], smoother: Callable[[RunKey, Any], Any]
) -> dict[RunKey, Any]:
    """Applies a sample-aware smoothing function to each loss history in a results dictionary."""
    return {
        run_key: smoother(run_key, loss_history)
        for run_key, result_obj in loss_dict.items()
        if (loss_history := get_loss_history_from_result(result_obj)) is not None
    }


@apply_to_list_of_dicts
def extract_noise_loss_dicts(
    loss_dict: dict[RunKey, Any], smoother: Callable[[RunKey, Any], Any]
) -> dict[RunKey, np.ndarray]:
    """Calculates the 'noise' for each loss history by subtracting a smoothed version."""
    noise_dict = {}
    for run_key, result_obj in loss_dict.items():
        loss_history = get_loss_history_from_result(result_obj)
        if loss_history is None:
            continue

        smoothed_history = smoother(run_key, loss_history)

        original = np.array(loss_history)
        smoothed = np.array(smoothed_history)

        if len(original) > len(smoothed):
            padding_width = len(original) - len(smoothed)
            smoothed = np.pad(smoothed, (padding_width, 0), mode="constant", constant_values=0)
        elif len(original) < len(smoothed):
            raise ValueError(
                f"Smoother for RunKey {run_key} produced a history of length {len(smoothed)}, "
                f"which is longer than the original length {len(original)}. "
                "Padding is only supported for shorter smoothed histories."
            )

        noise = original - smoothed
        noise_dict[run_key] = noise

    return noise_dict


def subsample_loss_dict_periodic(loss_dict: dict[RunKey, Any], subsample_by: str, every: int) -> dict[RunKey, Any]:
    if not loss_dict or every <= 0:
        return {}

    if subsample_by not in ("batch_size", "eta", "both"):
        raise ValueError("subsample_by must be either 'batch_size', 'eta', or 'both'")

    subsampled_bs = None
    if subsample_by in ("batch_size", "both"):
        all_bs = sorted(list({key.batch_size for key in loss_dict.keys()}))
        subsampled_bs = set(all_bs[::every])

    subsampled_etas = None
    if subsample_by in ("eta", "both"):
        all_etas = sorted(list({key.eta for key in loss_dict.keys()}))
        subsampled_etas = set(all_etas[::every])

    new_dict = {
        key: value
        for key, value in loss_dict.items()
        if (subsampled_bs is None or key.batch_size in subsampled_bs)
        and (subsampled_etas is None or key.eta in subsampled_etas)
    }

    return new_dict


def filter_experiments(
    experiments: dict[str, ExperimentBase],
    experiment_type: Type[ExperimentBase],
    loss_type: LossType | None = None,
    optimizer: OptimizerType | None = None,
) -> dict[str, ExperimentBase]:
    """
    Filters experiments based on experiment type, and optionally on loss type and/or optimizer.
    Loss type (resp. optimizers) defaults to MSE (resp None) for experiments that don't specify it.
    """

    def matches_criteria(experiment: ExperimentBase) -> bool:
        if not isinstance(experiment, experiment_type):
            return False

        if loss_type is not None:
            exp_loss_type = getattr(experiment, "loss_type", LossType.MSE)
            if exp_loss_type != loss_type:
                return False

        if optimizer is not None:
            exp_optimizer = getattr(experiment, "optimizer", None)
            if exp_optimizer != optimizer:
                return False

        return True

    return {name: exp for name, exp in experiments.items() if matches_criteria(exp)}


@apply_to_list_of_dicts
def filter_loss_dict_by_cutoff(
    loss_dict: dict[RunKey, Any],
    filter_by: str,
    cutoff: float,
    filter_below: bool = True,
) -> dict[RunKey, Any]:
    """Filters a dictionary by a cutoff value for a specified parameter."""
    if filter_by not in ("B", "eta", "temp"):
        raise ValueError("filter_by must be one of 'B', 'eta', or 'temp'")

    return {
        rk: val
        for rk, val in loss_dict.items()
        if (v := _get_run_key_value(rk, filter_by)) is not None and (v >= cutoff if filter_below else v <= cutoff)
    }


def extract_loss_histories(
    results_dict: dict[RunKey, Any],
) -> dict[RunKey, list[float]]:
    return {
        run_key: loss_history
        for run_key, result_obj in results_dict.items()
        if (loss_history := get_loss_history_from_result(result_obj)) is not None
    }
