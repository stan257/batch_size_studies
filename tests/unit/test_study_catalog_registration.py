from batch_size_studies.experiment_registry import clear_registry
from batch_size_studies.studies.catalog import (
    build_default_specs,
    linear_teacher_specs,
    mnist1m_sampled_specs,
    mnist1m_specs,
    register_default_studies,
)


def test_default_spec_builders_have_unique_names():
    specs = build_default_specs()
    names = [spec.name for spec in specs]
    assert specs
    assert len(names) == len(set(names))


def test_register_default_studies_is_idempotent():
    clear_registry()

    first = register_default_studies(force=True)
    assert first
    count_first = len(first)

    second = register_default_studies()
    assert len(second) == count_first


def test_backward_compatible_builder_names_remain_available():
    assert linear_teacher_specs()
    assert mnist1m_specs()
    assert mnist1m_sampled_specs()
