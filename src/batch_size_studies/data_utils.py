from functools import wraps
from typing import Any, Callable, Type, TypeVar

import numpy as np

from .definitions import LossType, OptimizerType, Parameterization, RunKey
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
            # For a trailing smoother, the smoothed history is shorter.
            # To align them, we drop the initial values from the original history
            # so that both arrays start at the first point where a full window is available.
            diff = len(original) - len(smoothed)
            original = original[diff:]
        elif len(original) < len(smoothed):
            raise ValueError(
                f"Smoother for RunKey {run_key} produced a history of length {len(smoothed)}, "
                f"which is longer than the original length {len(original)}. "
                "This is not supported."
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
    parameterization: Parameterization | None = None,
    optimizer: OptimizerType | None = None,
) -> dict[str, ExperimentBase]:
    """
    Filters experiments based on experiment type, and optionally on loss type and/or optimizer.
    Loss type (resp. optimizers) defaults to MSE (resp None) for experiments that don't specify it.
    """
    # Input validation
    if loss_type is not None and not isinstance(loss_type, LossType):
        raise TypeError(f"loss_type must be of type LossType, but got {type(loss_type).__name__}.")

    if parameterization is not None and not isinstance(parameterization, Parameterization):
        raise TypeError(
            f"parameterization must be of type Parameterization, but got {type(parameterization).__name__}."
        )

    if optimizer is not None and not isinstance(optimizer, OptimizerType):
        raise TypeError(f"optimizer must be of type OptimizerType, but got {type(optimizer).__name__}.")

    def matches_criteria(experiment: ExperimentBase) -> bool:
        if not isinstance(experiment, experiment_type):
            return False

        if loss_type is not None:
            exp_loss_type = getattr(experiment, "loss_type", LossType.MSE)
            if exp_loss_type != loss_type:
                return False

        if parameterization is not None:
            exp_parameterization = getattr(experiment, "parameterization")
            if exp_parameterization != parameterization:
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


@apply_to_list_of_dicts
def filter_loss_dict_by_loss_threshold(
    loss_dict: dict[RunKey, Any],
    threshold: float,
) -> dict[RunKey, Any]:
    """
    Filters a loss dictionary by removing runs where the loss exceeds a given
    threshold at any point in the history. Any non-finite loss (NaN, inf)
    will also cause the run to be removed.
    """
    return {
        rk: result_obj
        for rk, result_obj in loss_dict.items()
        if (loss_history := get_loss_history_from_result(result_obj)) is None
        or all(np.isfinite(loss) and loss <= threshold for loss in loss_history)
    }


def extract_loss_histories(
    results_dict: dict[RunKey, Any],
) -> dict[RunKey, list[float]]:
    return {
        run_key: loss_history
        for run_key, result_obj in results_dict.items()
        if (loss_history := get_loss_history_from_result(result_obj)) is not None
    }


def get_first_divergence_eta(divergent_runs: set[RunKey]) -> dict[int, float]:
    """
    Given a set of RunKeys for divergent runs, finds the smallest eta that
    caused divergence for each batch size.
    """
    first_divergence: dict[int, float] = {}
    for run_key in divergent_runs:
        if run_key.batch_size not in first_divergence or run_key.eta < first_divergence[run_key.batch_size]:
            first_divergence[run_key.batch_size] = run_key.eta
    return first_divergence


def adjust_run_keys_in_dict(
    input_dict: dict[RunKey, Any], adjuster_fn: Callable[[RunKey], RunKey]
) -> dict[RunKey, Any]:
    """
    Takes a dictionary with RunKey keys and returns a new dictionary
    with keys adjusted according to a provided callable.

    Args:
        input_dict (dict[RunKey, Any]): The input dictionary with RunKey keys.
        adjuster_fn (Callable[[RunKey], RunKey]): A callable that takes a RunKey
                                                  and returns an adjusted RunKey.
    Returns:
        dict[RunKey, Any]: A new dictionary with adjusted RunKey keys.
    """
    return {adjuster_fn(run_key): value for run_key, value in input_dict.items()}
