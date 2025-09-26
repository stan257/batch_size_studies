import optax

from .definitions import OptimizerType, Parameterization


def eta_adjustment_fn(experiment, eta: float):
    """
    The learning rate adjustment schedule is based on the findings from:
    "The Optimization Landscape of SGD Across the Feature Learning Strength"
    Atanasov et al. (2025, arXiv:2410.04642)

    For muP, this is scaled by the width N.
    """
    gamma = experiment.gamma
    depth = experiment.L

    base_lr = eta * gamma ** (2 / depth) if gamma > 1 else eta * gamma**2

    if experiment.parameterization == Parameterization.MUP:
        return base_lr * experiment.N
    return base_lr


def create_optimizer(experiment, eta: float):
    """
    Creates an optax optimizer based on the experiment configuration.
    Handles the eta adjustment for SGD (TODO: later ADAM) internally.
    """
    match experiment.optimizer:
        case OptimizerType.SGD:
            # The learning rate adjustment is specific to the SGD theory.
            learning_rate = eta_adjustment_fn(experiment, eta)
            return optax.sgd(learning_rate)
        case OptimizerType.ADAM:
            learning_rate = eta  # TODO: Update adjust the rate accordingly for ADAM
            return optax.adam(learning_rate)
        case _:
            raise NotImplementedError(f"Optimizer {experiment.optimizer} not implemented.")
