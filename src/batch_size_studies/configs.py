"""
Centralized Configuration for Experiments

This module is the single source of truth for all experiment definitions and
hyperparameter grids used in the project. Runner and reporting scripts should
import their configurations from here.
"""

import numpy as np

from .definitions import LossType, OptimizerType, Parameterization
from .experiments import (
    MNIST1MExperiment,
    MNIST1MSampledExperiment,
)


def get_main_hyperparameter_grids():
    batch_sizes = (2 ** np.arange(0, 17)).tolist()
    etas = np.power(2.0, np.arange(-12, 13)).tolist()
    return batch_sizes, etas


def get_main_experiment_configs():
    P = 100_000
    D = 25
    N = 100
    depth = 2
    K = 2
    NUM_STEPS = 1000

    gammas = [1e-5, 0.01, 0.1, 1.0, 10.0, 100.0, 1e5]

    experiments_to_run = {}

    # Polynomial teacher experiments
    kwargs_exp = dict(
        D=D,
        P=P,
        N=N,
        K=K,
        num_steps=NUM_STEPS,
        L=depth,
        parameterization=Parameterization.SP,
    )
    # for g in gammas:
    #     name = f"poly_gamma{str(g).replace('.', 'p')}_fixed_time"
    #     experiments_to_run[name] = SyntheticExperimentFixedTime(**(kwargs_exp | {"gamma": float(g)}))

    # MLP teacher experiments
    # mlp_teacher_kwargs = dict(
    #     D=D,
    #     P=P,
    #     N=N,
    #     L=depth,
    #     parameterization=Parameterization.SP,
    #     num_steps=NUM_STEPS,
    #     teacher_N=64,
    #     teacher_L=2,
    #     teacher_gamma=1.0,
    #     teacher_parameterization=Parameterization.SP,
    # )
    # for g in gammas:
    #     name = f"mlp_teacher_gamma{str(g).replace('.', 'p')}_fixed_time"
    #     experiments_to_run[name] = SyntheticExperimentMLPTeacher(**(mlp_teacher_kwargs | {"gamma": float(g)}))

    # # --- MNIST Experiment ---
    # # A single experiment definition for MNIST classification.
    # experiments_to_run["mnist_classification_mup"] = MNISTExperiment(
    #     N=512,
    #     L=2,
    #     num_epochs=1,
    #     parameterization=Parameterization.MUP,
    # )

    # --- MNIST-1M Experiment ---
    mnist1m_kwargs = dict(
        N=128,
        L=3,  # two hidden layers for this experiment type
        num_epochs=1,
        parameterization=Parameterization.MUP,  # we default to muP for experiments
    )
    # for opt in OptimizerType:
    opt = OptimizerType.ADAM
    # for loss_type in LossType:
    loss_type = LossType.MSE
    for g in gammas:
        name = f"mnist1m_mup_{loss_type.value}_{opt.value}_gamma{str(g).replace('.', 'p')}"
        experiments_to_run[name] = MNIST1MExperiment(
            **(
                mnist1m_kwargs
                | dict(
                    optimizer=opt,
                    loss_type=loss_type,
                    gamma=g,
                )
            )
        )

    return experiments_to_run


# --- Small muP Experiment Suite ---
# These are used to find smallest widths at which we start having μ-transfer across width


def get_small_mup_hyperparameter_grids():
    batch_sizes = (2 ** np.arange(7, 9)).tolist()
    etas = np.power(2.0, np.arange(4, 6)).tolist()
    return batch_sizes, etas


def get_small_mup_experiment_configs():
    P = 30_000
    D = 25
    K = 2
    depth = 3  # L=3 corresponds to 2 hidden layers
    NUM_STEPS = 100

    gammas = [0.01, 0.1, 1.0, 10.0, 100.0]
    widths = [64, 128, 256]

    experiments_to_run = {}

    # Synthetic muP experiments
    # for n_val in widths:
    #     for g in gammas:
    #         name = f"mup_L{depth}_N{n_val}_gamma{str(g).replace('.', 'p')}_fixed_time"
    #         experiments_to_run[name] = SyntheticExperimentFixedTime(
    #             D=D,
    #             P=P,
    #             N=n_val,
    #             K=K,
    #             num_steps=NUM_STEPS,
    #             L=depth,
    #             gamma=float(g),
    #             parameterization=Parameterization.MUP,
    #         )

    # MNIST muP experiments on sampled MNIST-1M
    depth_mnist = 3
    num_epochs = 1
    max_train_samples = 50_000  # Use a subset for faster runs
    for n_val in widths:
        for g in gammas:
            name = f"mnist1m_sampled_mup_L{depth_mnist}_N{n_val}_gamma{str(g).replace('.', 'p')}"
            experiments_to_run[name] = MNIST1MSampledExperiment(
                N=n_val,
                L=depth_mnist,
                num_epochs=num_epochs,
                gamma=float(g),
                parameterization=Parameterization.MUP,
                max_train_samples=max_train_samples,
                loss_type=LossType.MSE,
            )

    return experiments_to_run
