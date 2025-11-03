import importlib.util
import sys
from pathlib import Path

from batch_size_studies.definitions import LossType, OptimizerType, Parameterization
from batch_size_studies.experiments import SyntheticExperimentFixedTime

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
RUN_EXPERIMENTS_PATH = SCRIPTS_DIR / "run_experiments.py"

spec = importlib.util.spec_from_file_location("run_experiments", RUN_EXPERIMENTS_PATH)
run_experiments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiments)


def test_run_experiments_cli_smoke(monkeypatch, tmp_path):
    batch_sizes = [4]
    etas = [0.1]

    toy_experiment = SyntheticExperimentFixedTime(
        D=2,
        P=8,
        N=4,
        K=2,
        num_steps=3,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )

    monkeypatch.setattr(run_experiments, "get_main_experiment_configs", lambda: {"toy": toy_experiment})
    monkeypatch.setattr(run_experiments, "get_main_hyperparameter_grids", lambda: (batch_sizes, etas))
    monkeypatch.setattr(run_experiments, "setup_logging", lambda: None)

    recorded_calls = []

    def fake_run_single(*args, **kwargs):
        no_save_arg = kwargs.get("no_save")
        if no_save_arg is None and len(args) >= 6:
            no_save_arg = args[5]
        recorded_calls.append((args[0], no_save_arg))
        return args[0]

    monkeypatch.setattr(run_experiments, "run_single_experiment", fake_run_single)

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class DummyExecutor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, func, *args, **kwargs):
            return DummyFuture(func(*args, **kwargs))

    monkeypatch.setattr(run_experiments, "ProcessPoolExecutor", lambda: DummyExecutor())
    monkeypatch.setattr(run_experiments, "as_completed", lambda futures: futures)

    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--no-save", "--name", "toy"])
    run_experiments.main()

    assert recorded_calls == [("toy", True)]
