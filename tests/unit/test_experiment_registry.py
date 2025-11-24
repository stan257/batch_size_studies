from batch_size_studies.configs import get_main_experiment_configs, list_main_experiment_specs
from batch_size_studies.definitions import LossType, OptimizerType
from batch_size_studies.experiment_registry import list_registered_specs


def test_registry_matches_filter_conditions():
    specs = list_registered_specs()
    assert specs, "Registry should not be empty."

    target = next(
        spec for spec in specs if spec.optimizer == OptimizerType.SGD and spec.loss_type == LossType.MSE
    )
    filtered = [
        spec
        for spec in specs
        if spec.matches(
            experiment_types=[target.family],
            optimizer=target.optimizer,
            loss_type=target.loss_type,
        )
    ]
    assert filtered, "Filter should keep at least one spec."
    assert all(spec.family == target.family for spec in filtered)
    assert all(spec.optimizer == target.optimizer for spec in filtered)
    assert all(spec.loss_type == target.loss_type for spec in filtered)


def test_registered_specs_have_unique_names_and_buildable():
    specs = list_main_experiment_specs()
    assert specs, "Registry should contain at least one experiment spec."

    names = [spec.name for spec in specs]
    assert len(names) == len(set(names)), "Experiment spec names must be unique."

    for spec in specs:
        experiment = spec.build()
        assert experiment.experiment_type == spec.family


def test_main_configs_filter_by_optimizer_and_loss():
    configs = get_main_experiment_configs(
        experiment_types=["mnist1m_classification"],
        optimizer=OptimizerType.ADAM,
        loss_type=LossType.XENT,
    )
    assert configs, "Filter should return some configs."

    for name, exp in configs.items():
        assert "mnist1m" in name
        assert exp.optimizer == OptimizerType.ADAM
        assert exp.loss_type == LossType.XENT
