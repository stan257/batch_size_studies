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


def reverse_eta_adjustment_theoretical(
    func: Callable[[int], float], experiment, *, legacy_mse_scaling: bool = False
) -> Callable[[int], float]:
    """Returns the theoretical learning rate bound after undoing μP/γ scaling.

    If ``legacy_mse_scaling`` is True, an additional factor (2/num_outputs) is applied
    to reproduce the normalization used prior to the ½ ∥·∥² loss rescaling.
    """
    # Reverse all width and γ adjustments
    base_reversed_func = reverse_eta_adjustment(func, experiment)
    theoretical_divisor = 1.0

    # For legacy experiments that averaged MSE over outputs, account for the extra factor.
    if legacy_mse_scaling and experiment.loss_type == LossType.MSE:
        num_outputs = getattr(experiment, "num_outputs", 1)
        if num_outputs > 0:
            theoretical_divisor = 2.0 / num_outputs
        else:
            theoretical_divisor = 2.0

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
