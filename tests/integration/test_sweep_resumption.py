from dataclasses import replace

import numpy as np

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization
from batch_size_studies.experiments import SyntheticExperimentFixedData
from batch_size_studies.runner import run_experiment_sweep


def test_sweep_resumption_is_correct(tmp_path):
    """
    An integration test to verify that an interrupted and resumed
    sweep produces the exact same final results as an uninterrupted one.

    This test validates the interaction between the runner's state management,
    the checkpoint manager, and the trial runner's ability to resume.
    """
    # --- 1. Define a simple, deterministic experiment ---
    config = SyntheticExperimentFixedData(
        D=8,
        P=128,
        N=16,
        K=2,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
        seed=42,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
        num_epochs=4,  # Total epochs for the full run
    )
    batch_sizes = [32, 64]
    etas = [0.01, 0.001]
    init_key = 0

    # --- 2. Run the experiment without interruption to get the ground truth ---
    uninterrupted_results, uninterrupted_failures = run_experiment_sweep(
        experiment=config,
        batch_sizes=batch_sizes,
        etas=etas,
        init_key=init_key,
        directory=str(tmp_path / "uninterrupted"),
    )

    # --- 3. Simulate an interrupted run ---
    # Run the sweep for only 2 epochs. This will leave live checkpoints.
    interrupted_config = replace(config, num_epochs=2)
    run_experiment_sweep(
        experiment=interrupted_config,
        batch_sizes=batch_sizes,
        etas=etas,
        init_key=init_key,
        directory=str(tmp_path / "resumed"),
    )

    # --- 4. Run the full sweep again, which should resume from the checkpoints ---
    resumed_results, resumed_failures = run_experiment_sweep(
        experiment=config,  # Use the original 4-epoch config
        batch_sizes=batch_sizes,
        etas=etas,
        init_key=init_key,
        directory=str(tmp_path / "resumed"),
    )

    # --- 5. Assert that the final results are identical ---
    assert uninterrupted_failures == resumed_failures
    assert uninterrupted_results.keys() == resumed_results.keys()
    for key in uninterrupted_results:
        np.testing.assert_allclose(
            uninterrupted_results[key]["loss_history"],
            resumed_results[key]["loss_history"],
            err_msg=f"Loss history for {key} did not match after resumption.",
        )
