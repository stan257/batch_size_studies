import numpy as np
import optax

from .definitions import OptimizerType, Parameterization


def eta_adjustment_fn(experiment, eta: float):
    """
    The learning rate adjustment schedule is based on the findings from:
    "The Optimization Landscape of SGD Across the Feature Learning Strength"
    Atanasov et al. (2025, arXiv:2410.04642)

    For μP, we scale by the width N for SGD (resp. sqrt(N) for ADAM) to ensure
    μ-transfer across width.
    """
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
    base_lr = eta * gamma_mult * width_mult
    return base_lr


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
