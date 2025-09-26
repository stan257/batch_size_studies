from functools import wraps
from typing import Any, Callable, TypeVar

import numpy as np

from .definitions import RunKey

D = TypeVar("D", bound=dict)


def apply_to_list_of_dicts(func: Callable) -> Callable:
    """
    A decorator to allow a function that processes a single dictionary to
    transparently handle a list of dictionaries as well.
    """

    @wraps(func)
    def wrapper(data: D | list[D], *args, **kwargs) -> D | list[D]:
        if isinstance(data, list):
            return [func(item, *args, **kwargs) for item in data]
        # We've established it's not a list, so it must be a dict.
        # The type hint `D | list[D]` helps communicate this.
        return func(data, *args, **kwargs)

    return wrapper


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Computes the moving average of a 1D array using convolution."""
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def get_loss_history_from_result(result: Any) -> Any:
    """
    Robustly extracts a loss history list from a result object.
    Handles list, numpy/jax array, and dict-of-results formats.

    Args:
        result: The result object, which can be a list of losses, a numpy/jax
                array, or a dictionary containing a 'loss_history' key.

    Returns:
        A list or array representing the loss history, or None if not found.
    """
    if isinstance(result, dict):
        # Handles new format: {'loss_history': [...], 'other_metric': ...}
        return result.get("loss_history")
    if isinstance(result, (list, np.ndarray)):
        # Handles old format: [...] or np.array(...)
        return result
    # A simple check for other array-like objects (like JAX arrays)
    if hasattr(result, "shape") and hasattr(result, "dtype"):
        return result
    return None


def _get_run_key_value(rk: RunKey, by: str) -> float | int | None:
    """
    Extracts a value from a RunKey based on the 'by' parameter.
    """
    if by == "B":
        return rk.batch_size
    if by == "eta":
        return rk.eta
    if by == "temp":
        temp = rk.temp
        # Use rounding for floating point comparisons
        return round(temp, 8) if temp is not None else None
    return None


@apply_to_list_of_dicts
def filter_loss_dicts(loss_dict: dict, filter_by: str, values: list, keep: bool = True) -> dict:
    """
    Filters a dictionary by keeping or removing specified values for a given parameter.

    Args:
        loss_dict (dict): The dictionary to filter.
        filter_by (str): The parameter to filter on ('B', 'eta', or 'temp').
        values (list): The list of values to filter by.
        keep (bool): If True, keeps items with values in the list.
                     If False, removes them.
    """
    if filter_by not in ("B", "eta", "temp"):
        raise ValueError("filter_by must be one of 'B', 'eta', or 'temp'")

    # Round values for 'temp' to handle potential floating point inaccuracies
    value_set = {round(v, 8) if filter_by == "temp" else v for v in values}

    return {
        rk: val
        for rk, val in loss_dict.items()
        if (v := _get_run_key_value(rk, filter_by)) is not None and (v in value_set) == keep
    }


@apply_to_list_of_dicts
def uniform_smooth_loss_dicts(loss_dict: dict[RunKey, Any], smoother: Callable[[Any], Any]) -> dict[RunKey, Any]:
    """
    Applies a smoothing function to each loss history in a results dictionary.

    Returns a new dictionary mapping each RunKey to its smoothed loss history.
    Runs without a valid loss history are omitted.

    This function is decorated to also handle a list of dictionaries.
    """
    return {
        run_key: smoother(loss_history)
        for run_key, result_obj in loss_dict.items()
        if (loss_history := get_loss_history_from_result(result_obj)) is not None
    }


@apply_to_list_of_dicts
def sample_aware_smooth_loss_dicts(
    loss_dict: dict[RunKey, Any], smoother: Callable[[RunKey, Any], Any]
) -> dict[RunKey, Any]:
    """
    Applies a smoothing function to each loss history in a results dictionary,
    where the smoother also receives the RunKey.

    Returns a new dictionary mapping each RunKey to its smoothed loss history.
    Runs without a valid loss history are omitted.

    This function is decorated to also handle a list of dictionaries.
    """
    return {
        run_key: smoother(run_key, loss_history)
        for run_key, result_obj in loss_dict.items()
        if (loss_history := get_loss_history_from_result(result_obj)) is not None
    }


@apply_to_list_of_dicts
def extract_noise_loss_dicts(
    loss_dict: dict[RunKey, Any], smoother: Callable[[RunKey, Any], Any]
) -> dict[RunKey, np.ndarray]:
    """
    Calculates the 'noise' for each loss history by subtracting a smoothed version.

    The noise is defined as the element-wise difference between the original
    loss history and the output of the provided smoother function. If the
    smoothed history is shorter than the original, it is padded with zeros
    at the beginning to match the length before subtraction.

    Args:
        loss_dict: A dictionary mapping RunKey to a results object.
        smoother: A callable that takes a RunKey and a loss history and returns
                  a smoothed version of the history.

    Returns:
        A dictionary mapping each RunKey to a numpy array representing the
        noise. Runs without a valid loss history are omitted.

    Raises:
        ValueError: If the smoother returns a history that is longer than the
                    original history for any run.
    """
    noise_dict = {}
    for run_key, result_obj in loss_dict.items():
        loss_history = get_loss_history_from_result(result_obj)
        if loss_history is None:
            continue

        smoothed_history = smoother(run_key, loss_history)

        original = np.array(loss_history)
        smoothed = np.array(smoothed_history)

        if len(original) > len(smoothed):
            # Pad the beginning of the smoothed array with zeros to match length.
            # This is useful for 'valid' convolutions that shorten the signal.
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

    # Create the new dictionary with only the keys that match the subsampled values
    new_dict = {
        key: value
        for key, value in loss_dict.items()
        if (subsampled_bs is None or key.batch_size in subsampled_bs)
        and (subsampled_etas is None or key.eta in subsampled_etas)
    }

    return new_dict


@apply_to_list_of_dicts
def filter_loss_dict_by_cutoff(
    loss_dict: dict[RunKey, Any],
    filter_by: str,
    cutoff: float,
    filter_below: bool = True,
) -> dict[RunKey, Any]:
    """
    Filters a dictionary by a cutoff value for a specified parameter.

    Args:
        loss_dict (dict): The dictionary to filter.
        filter_by (str): The parameter to filter on ('B', 'eta', or 'temp').
        cutoff (float): The threshold value.
        filter_below (bool): If True (default), removes entries with values
            *below* the cutoff. If False, removes entries with values
            *above* the cutoff.

    This function is decorated to also handle a list of dictionaries.
    """
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
    """
    Extracts 'loss_history' lists from a results dictionary into a new dictionary.

    Args:
        results_dict: A dictionary mapping RunKey to a results object.
                      The result object can be a dictionary containing a
                      'loss_history' key, or a list of losses.

    Returns:
        A dictionary mapping each RunKey to its corresponding loss history list.
        Runs without a valid loss history are omitted.
    """
    return {
        run_key: loss_history
        for run_key, result_obj in results_dict.items()
        if (loss_history := get_loss_history_from_result(result_obj)) is not None
    }
