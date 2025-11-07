from batch_size_studies.configs import list_main_experiment_specs


def test_registered_specs_have_unique_names_and_buildable():
    specs = list_main_experiment_specs()
    assert specs, "Registry should contain at least one experiment spec."

    names = [spec.name for spec in specs]
    assert len(names) == len(set(names)), "Experiment spec names must be unique."

    for spec in specs:
        experiment = spec.build()
        assert experiment.experiment_type == spec.family
