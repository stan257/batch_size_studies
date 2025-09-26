import os
import pickle
from functools import partial

import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import LossType, Parameterization, RunKey
from batch_size_studies.experiments import MNIST1MSampledExperiment, SyntheticExperimentFixedData
from batch_size_studies.mnist_training import load_mnist1m_dataset, run_mnist_experiment
from batch_size_studies.synthetic_training import run_experiment as run_synthetic_experiment

# Location to store golden data files.
GOLDEN_DATA_DIR = os.path.join(os.path.dirname(__file__), "golden_data")


def compare_results(golden: dict, current: dict):
    """Asserts that two experiment data dictionaries are identical to machine precision."""
    # Compare failed runs
    assert golden["failed"] == current["failed"], "Set of failed runs does not match."

    # Compare results dict (loss histories etc)
    golden_results = golden["results"]
    current_results = current["results"]
    assert golden_results.keys() == current_results.keys(), "RunKeys in results do not match."

    for run_key in golden_results:
        golden_run = golden_results[run_key]
        current_run = current_results[run_key]

        # Compare loss history
        np.testing.assert_allclose(
            golden_run["loss_history"],
            current_run["loss_history"],
            rtol=1e-6,
            err_msg=f"Loss history mismatch for {run_key}",
        )
        # Compare other metrics if they exist
        if "epoch_test_accuracies" in golden_run:
            np.testing.assert_allclose(
                golden_run["epoch_test_accuracies"],
                current_run["epoch_test_accuracies"],
                rtol=1e-6,
                err_msg=f"Accuracy mismatch for {run_key}",
            )

    # Compare weight history
    golden_weights = golden["weights"]
    current_weights = current["weights"]
    assert golden_weights.keys() == current_weights.keys(), "Weight snapshot steps do not match."

    for step in golden_weights:
        golden_params = golden_weights[step]
        current_params = current_weights[step]
        # params are lists of jnp.ndarray
        assert len(golden_params) == len(current_params), f"Layer count mismatch at step {step}"
        for i in range(len(golden_params)):
            np.testing.assert_allclose(
                golden_params[i],
                current_params[i],
                rtol=1e-6,
                err_msg=f"Weight mismatch at step {step}, layer {i}",
            )


def run_and_get_all_data(config, batch_sizes, etas, tmp_path, **kwargs) -> dict:
    """A helper to run an experiment and load all its generated data."""
    if isinstance(config, SyntheticExperimentFixedData):
        run_synthetic_experiment(
            experiment=config, batch_sizes=batch_sizes, etas=etas, directory=str(tmp_path), num_epochs=2
        )
    elif isinstance(config, MNIST1MSampledExperiment):
        run_mnist_experiment(
            experiment=config,
            batch_sizes=batch_sizes,
            etas=etas,
            directory=str(tmp_path),
            init_key=42,
            dataset_loader=kwargs.get("dataset_loader"),
        )
    else:
        raise TypeError(f"Unsupported config type for reproducibility test: {type(config)}")

    # After running, load the results and weights
    results_dict, failed_runs = config.load_results(directory=str(tmp_path))

    manager = CheckpointManager(config, directory=str(tmp_path))
    run_key = RunKey(batch_size=batch_sizes[0], eta=etas[0])
    weight_history = manager.load_full_weight_history(run_key)

    return {"results": results_dict, "failed": failed_runs, "weights": weight_history}


def get_synthetic_config():
    return SyntheticExperimentFixedData(
        D=10, P=1024, N=16, K=2, gamma=1.0, L=2, parameterization=Parameterization.SP, seed=42
    )


def get_mnist_config():
    return MNIST1MSampledExperiment(
        N=16, L=2, parameterization=Parameterization.MUP, num_epochs=2, max_train_samples=4096, loss_type=LossType.MSE
    )


def _run_reproducibility_test(config_fn, golden_filename, tmp_path, **kwargs):
    config = config_fn()
    batch_sizes = [32]
    etas = [0.01]
    golden_filepath = os.path.join(GOLDEN_DATA_DIR, golden_filename)

    if os.environ.get("REGENERATE_GOLDEN_DATA"):
        print(f"\n--- Regenerating golden data for: {golden_filename} ---")
        os.makedirs(GOLDEN_DATA_DIR, exist_ok=True)
        data = run_and_get_all_data(config, batch_sizes, etas, tmp_path, **kwargs)
        with open(golden_filepath, "wb") as f:
            pickle.dump(data, f)
        pytest.skip(f"Golden data regenerated at {golden_filepath}. Skipping test.")

    if not os.path.exists(golden_filepath):
        pytest.fail(
            f"Golden data file not found: {golden_filepath}. Run with 'REGENERATE_GOLDEN_DATA=1 pytest' to create it."
        )

    # Run the experiment and get current data
    current_data = run_and_get_all_data(config, batch_sizes, etas, tmp_path, **kwargs)

    # Load golden data
    with open(golden_filepath, "rb") as f:
        golden_data = pickle.load(f)

    # Compare
    compare_results(golden_data, current_data)


def test_synthetic_reproducibility(tmp_path):
    """
    Tests that the synthetic experiment training is perfectly reproducible.
    """
    _run_reproducibility_test(get_synthetic_config, "synthetic_golden.pkl", tmp_path)


def test_mnist_reproducibility(tmp_path, fake_mnist1m_data_dir):
    """
    Tests that the MNIST experiment training is perfectly reproducible.
    """
    loader = partial(load_mnist1m_dataset, data_dir=fake_mnist1m_data_dir)
    _run_reproducibility_test(get_mnist_config, "mnist_golden.pkl", tmp_path, dataset_loader=loader)
