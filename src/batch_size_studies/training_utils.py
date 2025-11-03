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


def create_base_optimizer_transform(optimizer_type: OptimizerType):
    """
    Creates the part of the optimizer transform before learning rate scaling.
    """
    match optimizer_type:
        case OptimizerType.SGD:
            # For basic SGD, there's no stateful transform before scaling.
            return optax.identity()
        case OptimizerType.ADAM:
            # For Adam, it's the scale_by_adam transform.
            return optax.scale_by_adam()
        case _:
            raise NotImplementedError(f"Optimizer {optimizer_type} not implemented.")
