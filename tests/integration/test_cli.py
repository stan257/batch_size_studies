import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

# =============================================================================
# Load the script as a module to test its main() function
# =============================================================================
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
RUN_EXPERIMENTS_PATH = SCRIPTS_DIR / "run_experiments.py"

spec = importlib.util.spec_from_file_location("run_experiments", RUN_EXPERIMENTS_PATH)
run_experiments_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiments_module)


def test_run_experiments_cli_smoke(monkeypatch):
    """
    Tests that the CLI script correctly parses arguments and calls the
    core orchestration function in the runner module.
    """
    # 1. Patch the dependencies of the script module
    mock_run_from_args = MagicMock()
    # Patch the name 'run_from_cli_args' where it is looked up (in the script's module)
    monkeypatch.setattr(run_experiments_module, "run_from_cli_args", mock_run_from_args)
    monkeypatch.setattr(run_experiments_module, "setup_logging", lambda: None)

    # 2. Simulate command-line arguments
    sys.argv = ["run_experiments.py", "run", "--no-save", "--name", "toy_experiment"]

    # 3. Run the script's main function
    run_experiments_module.main()

    # 4. Assert that the core logic function was called correctly
    mock_run_from_args.assert_called_once()
    called_args = mock_run_from_args.call_args[0][0]

    assert called_args.command == "run"
    assert called_args.name == ["toy_experiment"]
    assert called_args.no_save is True
