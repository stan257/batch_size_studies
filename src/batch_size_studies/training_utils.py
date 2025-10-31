import functools
from typing import Callable

import optax

from .definitions import LossType, OptimizerType


def reverse_eta_adjustment(func: Callable[[int], float], experiment) -> Callable[[int], float]:
    """
    This "undoes" the scaling applied by `eta_adjustment_fn`.
    """
    adj_factor = experiment.get_adjusted_eta(1.0)

    if adj_factor == 0:
        return lambda b: float("inf")

    @functools.wraps(func)
    def reversed_func(batch_size: int) -> float:
        eta_eff_bound = func(batch_size)
        return eta_eff_bound / adj_factor

    return reversed_func


def reverse_eta_adjustment_theoretical(func: Callable[[int], float], experiment) -> Callable[[int], float]:
    """
    Returns effective learning rate to match the theory. This includes
    - reversion width and γ-adjustments (for μP)
    - divide by 2 to match 1/2 * E[Loss] results
    - (for classification w/ MSE) further adjusts loss f-n to account for the one hot encoding coming with MSE

    This is useful for comparing empirical stability bounds with theoretical predictions.
    """
    # Reverse all width and γ adjustments
    base_reversed_func = reverse_eta_adjustment(func, experiment)
    # always divide by 2 to match theory
    theoretical_divisor = 2

    # for MSE and classification tasks adjsut by num targets
    if experiment.loss_type == LossType.MSE:
        num_outputs = getattr(experiment, "num_outputs", 1)
        theoretical_divisor = 2 / num_outputs

    @functools.wraps(func)
    def theoretical_reversed_func(batch_size: int) -> float:
        # Get the eta bound with standard adjustments undone
        eta_bound = base_reversed_func(batch_size)
        # Apply the additional theoretical scaling
        return eta_bound / theoretical_divisor

    return theoretical_reversed_func


def create_optimizer(experiment, learning_rate: float):
    """
    Creates an optax optimizer based on the experiment configuration.
    """
    match experiment.optimizer:
        case OptimizerType.SGD:
            return optax.sgd(learning_rate)
        case OptimizerType.ADAM:
            return optax.adam(learning_rate)
        case _:
            raise NotImplementedError(f"Optimizer {experiment.optimizer} not implemented.")
