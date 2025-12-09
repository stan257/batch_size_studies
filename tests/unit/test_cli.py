from types import SimpleNamespace

import batch_size_studies.cli as cli_module


class DummyConfig:
    def __init__(self, name):
        self.name = name
        self.experiment_type = "dummy_type"

    def get_filepath(self, directory):
        return f"{directory}/{self.name}.pkl"


def make_args(**overrides):
    defaults = dict(
        override=None,
        dry_run=False,
        dry_run_steps=5,
        no_save=True,
        eta_stability_depth=None,
        max_eval_samples=None,
        save_interstitial_snapshots=None,
        save_epoch_snapshots=None,
        num_processes=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_run_planner_creates_plan_with_tasks():
    args = make_args(no_save=True)
    configs = {"exp_a": DummyConfig("exp_a"), "exp_b": DummyConfig("exp_b")}
    planner = cli_module.RunPlanner(args, configs, batch_sizes=[32], etas=[0.1], directory="expdir")

    plan = planner.build()

    assert plan is not None
    assert len(plan.tasks) == 2
    assert plan.tasks[0].name in configs
    assert plan.no_save is True
    assert plan.batch_sizes == [32]


def test_run_planner_builds_dry_run_plan():
    args = make_args(dry_run=True, dry_run_steps=7)
    configs = {"exp": DummyConfig("exp")}
    planner = cli_module.RunPlanner(args, configs, batch_sizes=[16], etas=[0.2], directory="expdir")

    plan = planner.build()

    assert plan is not None
    assert plan.dry_run_task is not None
    assert plan.dry_run_batch == 16
    assert plan.dry_run_eta == 0.2
    assert plan.dry_run_steps == 7
    assert plan.tasks == []


def test_run_executor_dry_run(monkeypatch):
    captured = {}

    def fake_run_experiment_sweep(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "run_experiment_sweep", fake_run_experiment_sweep)

    plan = cli_module.RunPlan(
        directory="expdir",
        batch_sizes=[8],
        etas=[0.05],
        tasks=[],
        no_save=True,
        eta_stability_depth=None,
        max_eval_samples=None,
        save_interstitial_snapshots=None,
        save_epoch_snapshots=None,
        num_processes=1,
        dry_run_task=cli_module.RunTask("exp", DummyConfig("exp")),
        dry_run_batch=8,
        dry_run_eta=0.05,
        dry_run_steps=3,
    )

    executor = cli_module.RunExecutor()
    executor.run_dry(plan)

    assert captured["batch_sizes"] == [8]
    assert captured["etas"] == [0.05]
    assert captured["dry_run_steps"] == 3
    assert captured["dry_run"] is True


def test_run_executor_sequential(monkeypatch):
    calls = []

    def fake_run_single(name, config, *args, **kwargs):
        calls.append(name)

    monkeypatch.setattr(cli_module, "_run_single_experiment", fake_run_single)

    plan = cli_module.RunPlan(
        directory="expdir",
        batch_sizes=[4],
        etas=[0.1],
        tasks=[cli_module.RunTask("exp_a", DummyConfig("exp_a")), cli_module.RunTask("exp_b", DummyConfig("exp_b"))],
        no_save=False,
        eta_stability_depth=None,
        max_eval_samples=None,
        save_interstitial_snapshots=None,
        save_epoch_snapshots=None,
        num_processes=1,
    )

    executor = cli_module.RunExecutor()
    executor.execute(plan)

    assert calls == ["exp_a", "exp_b"]


def test_custom_unpickler_remaps_legacy_classes(monkeypatch):
    import io

    from batch_size_studies.storage_utils import _EXPERIMENT_CLASS_REMAP, _LEGACY_EXPERIMENT_MODULE, CustomUnpickler

    for legacy_class, new_module in _EXPERIMENT_CLASS_REMAP.items():
        unpickler = CustomUnpickler(io.BytesIO(b""))
        cls = unpickler.find_class(_LEGACY_EXPERIMENT_MODULE, legacy_class)
        assert cls.__module__.startswith(new_module)
