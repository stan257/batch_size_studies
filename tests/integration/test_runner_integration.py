import os
import pickle

from batch_size_studies.definitions import Parameterization, RunKey
from batch_size_studies.experiments import SyntheticExperimentFixedData
from batch_size_studies.runner import run_experiment_sweep


class TestRunnerIntegration:
    def test_end_to_end_file_saving_logic(self, tmp_path):
        """
        An integration test to verify the file saving logic from runner to storage.

        This test ensures that:
        1. `run_experiment_sweep` correctly orchestrates the run.
        2. `validate_and_store_result` calls `experiment.save_results`.
        3. `experiment.save_results` correctly generates the filepath.
        4. `storage_utils.save_experiment` writes the data to the correct path.
        """
        config = SyntheticExperimentFixedData(
            D=8, P=64, N=16, K=2, gamma=1.0, L=2, parameterization=Parameterization.SP, seed=42
        )

        results, failures = run_experiment_sweep(
            experiment=config,
            batch_sizes=[32],
            etas=[0.01],
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
        assert RunKey(32, 0.01) in saved_data["losses"]
