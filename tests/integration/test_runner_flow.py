import os
import pickle
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

import batch_size_studies.runner as runner_module
from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import SyntheticExperimentFixedData
from batch_size_studies.runner import run_experiment_sweep


class DummyTqdm:
    def __init__(self, iterable=None, total=None, **kwargs):
        self.iterable = iterable if iterable is not None else range(total or 0)

    def __iter__(self):
        return iter(self.iterable)

    def update(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class TinyModel:
    def __call__(self, params, inputs):
        return jnp.matmul(inputs, params["w"]) + params["b"]

    def init_params(self, init_key: int, widths: list[int]):
        w_key, b_key = jax.random.split(jax.random.PRNGKey(init_key))
        return {
            "w": jax.random.normal(w_key, (widths[0], widths[1])),
            "b": jax.random.normal(b_key, (widths[1],)),
        }


class TinyExperiment(SyntheticExperimentFixedData):
    def create_model_instance(self):
        return TinyModel()

    def get_model_widths(self) -> list[int]:
        return [self.D, 1]

    def get_model_wrapper(self, model_instance, params0):
        return model_instance


class DummyCheckpointManager:
    def __init__(self, directory):
        self.directory = directory
        self.metadata_saved = []

    def save_sweep_metadata(self, metadata):
        self.metadata_saved.append(metadata)

    def cleanup_live_checkpoint(self, run_key):
        pass


class TestRunnerIntegration:
    def test_end_to_end_file_saving_logic(self, tmp_path):
        """An integration test to verify the full lifecycle of a single trial.
        An integration test to verify the full lifecycle of a single trial.

        This test ensures that:
        1. `run_experiment_sweep` correctly orchestrates the run.
        2. The final results file is created and contains the correct data.
        3. The analysis weights file is created.
        4. The temporary live checkpoint file is cleaned up after a successful run.
        """
        config = SyntheticExperimentFixedData(
            D=8,
            P=64,
            N=16,
            K=2,
            gamma=1.0,
            L=2,
            parameterization=Parameterization.SP,
            seed=42,
            optimizer=OptimizerType.SGD,
            loss_type=LossType.MSE,
        )

        run_key = RunKey(batch_size=32, eta=0.01)

        results, failures = run_experiment_sweep(
            experiment=config,
            batch_sizes=[run_key.batch_size],
            etas=[run_key.eta],
            directory=str(tmp_path),
            num_epochs=1,  # Keep it short
        )

        expected_dir = tmp_path / config.experiment_type
        expected_filename = config.generate_filename()
        expected_filepath = expected_dir / expected_filename

        assert os.path.exists(expected_filepath), f"Expected results file not found at {expected_filepath}"

        with open(expected_filepath, "rb") as f:
            saved_data = pickle.load(f)

        assert saved_data["losses"] == results
        assert saved_data["failed_runs"] == failures
        assert len(saved_data["losses"]) == 1  # one run
        assert run_key in saved_data["losses"]

        # --- 2. Verify Analysis Weights and Live Checkpoint Files ---
        # Use CheckpointManager to derive the expected paths, just like the runner does.
        cm = CheckpointManager(config, directory=str(tmp_path))

        # The analysis weights file should exist.
        assert os.path.exists(cm.weights_filepath), "Analysis weights file was not created."

        # The live checkpoint file for this run should have been cleaned up.
        resume_filepath = cm._get_resume_filepath(run_key)
        assert not os.path.exists(resume_filepath), "Live checkpoint file was not cleaned up after successful run."

    @patch("batch_size_studies.runner.run_single_trial")
    def test_eta_stability_search_stops_early(self, mock_run_single_trial, tmp_path):
        # 1. Setup
        # This mock will always report success to trigger the stability tracker.
        mock_run_single_trial.return_value = True

        config = SyntheticExperimentFixedData(
            D=8,
            P=64,
            N=16,
            K=2,
            gamma=1.0,
            L=2,
            parameterization=Parameterization.SP,
            seed=42,
            optimizer=OptimizerType.SGD,
            loss_type=LossType.MSE,
        )

        # The runner sorts etas descending, so this is the order of execution.
        etas = [8.0, 4.0, 2.0, 1.0, 0.5]
        eta_stability_search_depth = 3

        # 2. Execution
        run_experiment_sweep(
            experiment=config,
            batch_sizes=[32],
            etas=etas,
            directory=str(tmp_path),
            eta_stability_search_depth=eta_stability_search_depth,
        )

        # 3. Assertions
        # The sweep should run for eta=8.0, 4.0, and 2.0 (3 successful runs).
        # It should then stop and not run for eta=1.0 or 0.5.
        assert mock_run_single_trial.call_count == eta_stability_search_depth

        # Check which etas were actually run by inspecting the mock's call arguments.
        called_etas = {call.kwargs["context"].run_key.eta for call in mock_run_single_trial.call_args_list}
        expected_run_etas = {8.0, 4.0, 2.0}
        assert called_etas == expected_run_etas


class TestRunExperimentSweepNoSaveIntegration:
    @pytest.fixture(autouse=True)
    def patch_tqdm(self, monkeypatch):
        monkeypatch.setattr(runner_module, "tqdm", DummyTqdm)

    def make_experiment(self):
        return TinyExperiment(
            D=2,
            P=4,
            N=2,
            K=1,
            num_epochs=1,
            seed=0,
            gamma=1.0,
            L=2,
            parameterization=Parameterization.SP,
            optimizer=OptimizerType.SGD,
            loss_type=LossType.MSE,
        )

    def test_single_run_success(self, tmp_path):
        experiment = self.make_experiment()

        results, failed = run_experiment_sweep(
            experiment=experiment,
            batch_sizes=[2],
            etas=[0.1],
            init_key=0,
            directory=str(tmp_path),
            no_save=True,
        )

        run_key = RunKey(batch_size=2, eta=0.1)
        assert run_key in results
        run_result = results[run_key]
        assert "loss_history" in run_result
        assert (
            len(run_result["loss_history"])
            == experiment.compute_num_steps(2, experiment.prepare_datasets(0)[0], None)[0]
        )
        assert run_key not in failed

    def test_skips_completed_run(self, tmp_path, monkeypatch):
        experiment = self.make_experiment()
        run_key = RunKey(batch_size=2, eta=0.1)
        pre_results = {run_key: {"loss_history": [0.1, 0.1]}}

        dummy_cm = DummyCheckpointManager(str(tmp_path))

        def fake_setup(experiment, directory, no_save, init_key):
            model_instance = experiment.create_model_instance()
            params0 = model_instance.init_params(init_key, experiment.get_model_widths())
            model_wrapper = experiment.get_model_wrapper(model_instance, params0)
            return pre_results.copy(), set(), dummy_cm, params0, model_wrapper

        run_calls = {"count": 0}

        def fake_run_single_trial(*args, **kwargs):
            run_calls["count"] += 1
            return True

        monkeypatch.setattr(runner_module, "_setup_sweep_state", fake_setup)
        monkeypatch.setattr(runner_module, "run_single_trial", fake_run_single_trial)

        results, failed = run_experiment_sweep(
            experiment=experiment,
            batch_sizes=[2],
            etas=[0.1],
            init_key=0,
            directory=str(tmp_path),
            no_save=False,
        )

        assert run_calls["count"] == 0
        assert results == pre_results
        assert failed == set()
