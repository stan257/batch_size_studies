import functools
from typing import Callable

import numpy as np
import optax

from .definitions import LossType, OptimizerType, Parameterization
from .experiments import MLPStudentExperiment


def get_eta_adjustment_factor(experiment) -> float:
    """
    The learning rate adjustment schedule is based on the findings from:
    "The Optimization Landscape of SGD Across the Feature Learning Strength"
    Atanasov et al. (2025, arXiv:2410.04642)

    For μP, we scale by the width N for SGD (resp. sqrt(N) for ADAM) to ensure
    μ-transfer across width.
    """
    if not isinstance(experiment, MLPStudentExperiment):
        # Only MLP students have this complex adjustment logic.
        # Linear models or others use a factor of 1.0.
        return 1.0

    gamma = experiment.gamma
    depth = experiment.L
    width = experiment.N

    match experiment.optimizer:
        case OptimizerType.SGD:
            gamma_mult = gamma ** (2 / depth) if gamma > 1 else gamma**2
        case OptimizerType.ADAM:
            gamma_mult = gamma ** (1 / depth) if gamma > 1 else gamma
        case _:
            # Default to returning the base eta if no specific rule is defined.
            gamma_mult = 1.0

    if experiment.parameterization == Parameterization.SP:
        width_mult = 1.0
    else:
        match experiment.optimizer:
            case OptimizerType.SGD:
                width_mult = width
            case OptimizerType.ADAM:
                width_mult = np.sqrt(width)
            case _:
                width_mult = 1.0

    return gamma_mult * width_mult


def eta_adjustment_fn(experiment, eta: float):
    adj_factor = get_eta_adjustment_factor(experiment)
    return eta * adj_factor


def reverse_eta_adjustment(func: Callable[[int], float], experiment) -> Callable[[int], float]:
    """
    This "undoes" the scaling applied by `eta_adjustment_fn`.
    """
    adj_factor = get_eta_adjustment_factor(experiment)

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


def create_optimizer(experiment, eta: float):
    """
    Creates an optax optimizer based on the experiment configuration.
    The learning rate is determined by `eta_adjustment_fn`.
    """
    learning_rate = eta_adjustment_fn(experiment, eta)
    match experiment.optimizer:
        case OptimizerType.SGD:
            return optax.sgd(learning_rate)
        case OptimizerType.ADAM:
            return optax.adam(learning_rate)
        case _:
            raise NotImplementedError(f"Optimizer {experiment.optimizer} not implemented.")
