"""
Centralized Configuration for Experiments

This module is the single source of truth for all experiment definitions and
hyperparameter grids used in the project. Runner and reporting scripts should
import their configurations from here.
"""

import numpy as np

from .definitions import LossType, OptimizerType, Parameterization
from .experiments import MNIST1MExperiment


def get_main_hyperparameter_grids():
    batch_sizes = (2 ** np.arange(0, 17)).tolist()
    etas = np.power(2.0, np.arange(-12, 14)).tolist()
    # etas = [3200.0, 3800.0, 4400.0, 5000.0]
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
    for opt in OptimizerType:
        # opt = OptimizerType.SGD
        for loss_type in LossType:
            # loss_type = LossType.XENT
            for g in gammas:
                name = f"mnist1m_mup_{loss_type.value}_{opt.value}_gamma{str(g).replace('.', 'p')}_epochs{mnist1m_kwargs['num_epochs']}"
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

    # --- MNIST-1M Sampled Experiment ---
    # mnist1m_sampled_kwargs = dict(
    #     N=128,
    #     L=3,
    #     num_epochs=20,
    #     parameterization=Parameterization.MUP,
    #     max_train_samples=65_536,
    # )
    # # for opt in OptimizerType:
    # opt = OptimizerType.SGD
    # # for loss_type in LossType:
    # loss_type = LossType.MSE
    # for g in gammas:
    #     name = f"mnist1m_sampled_mup_{loss_type.value}_{opt.value}_gamma{str(g).replace('.', 'p')}"
    #     experiments_to_run[name] = MNIST1MSampledExperiment(
    #         **(
    #             mnist1m_sampled_kwargs
    #             | dict(
    #                 optimizer=opt,
    #                 loss_type=loss_type,
    #                 gamma=g,
    #             )
    #         )
    #     )

    # --- Linear Teacher Experiments ---
    # linear_teacher_kwargs = dict(
    #     D=500,
    #     optimizer=OptimizerType.SGD,
    #     loss_type=LossType.MSE,
    # )

    # linear_teacher_online_kwargs = linear_teacher_kwargs | dict(P=100_000, num_epochs=1)
    # linear_teacher_offline_kwargs = linear_teacher_kwargs | dict(P=10_000, num_epochs=10)
    # linear_teacher_offline_longest_kwargs = linear_teacher_kwargs | dict(P=1_000, num_epochs=100)
    # # Experiment 1: alpha=1.1, beta=0.25
    # # alpha_beta_dict = dict(alpha=1.1, beta=0.25)
    # # name = "online_linear_teacher_alpha1p1_beta0p25_long"
    # # experiments_to_run[name] = SyntheticExperimentLinearTeacher(**(linear_teacher_online_kwargs | alpha_beta_dict))

    # # name = "offline_linear_teacher_alpha1p1_beta0p25_long"
    # # experiments_to_run[name] = SyntheticExperimentLinearTeacher(**(linear_teacher_offline_kwargs | alpha_beta_dict))

    # alpha_beta_dict = dict(alpha=2.0, beta=0.25)
    # name = "online_linear_teacher_alpha2p0_beta0p25_long"
    # experiments_to_run[name] = SyntheticExperimentLinearTeacher(**(linear_teacher_online_kwargs | alpha_beta_dict))

    # name = "offline_linear_teacher_alpha2p0_beta0p25_long"
    # experiments_to_run[name] = SyntheticExperimentLinearTeacher(**(linear_teacher_offline_kwargs | alpha_beta_dict))

    # name = "offline_linear_teacher_alpha2p0_beta0p25_longest"
    # experiments_to_run[name] = SyntheticExperimentLinearTeacher(
    #     **(linear_teacher_offline_longest_kwargs | alpha_beta_dict)
    # )

    # name = "offline_linear_teacher_alpha2p0_beta0p25_longest_fr"
    # experiments_to_run[name] = SyntheticExperimentLinearTeacher(
    #     **(linear_teacher_kwargs | dict(P=100, num_epochs=1000) | alpha_beta_dict)
    # )
    return experiments_to_run
