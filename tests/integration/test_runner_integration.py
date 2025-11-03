import os
import pickle
from unittest.mock import patch

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import SyntheticExperimentFixedData
from batch_size_studies.runner import run_experiment_sweep


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
        """
        Tests that the eta stability search correctly skips etas after
        'depth' consecutive successful runs.
        """
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
