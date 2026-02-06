import numpy as np

from batch_size_studies.configs import list_main_experiment_specs
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization
from batch_size_studies.experiment_types.mnist import MNIST1MExperiment, MNIST1MSampledExperiment


def _make_mnist1m(max_train_samples=None):
    return MNIST1MExperiment(
        N=32,
        L=2,
        parameterization=Parameterization.MUP,
        gamma=1.0,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
        num_epochs=2,
        max_train_samples=max_train_samples,
    )


def test_mnist1m_unsampled_defaults():
    exp = _make_mnist1m(max_train_samples=None)
    assert exp.experiment_type == "mnist1m_classification"
    assert exp.get_sweep_metadata(init_key=11) == {}
    assert not exp.should_skip_batch_size(batch_size=128, train_ds=None)
    assert "max_train_samples" not in exp.to_params_dict()


def test_mnist1m_sampled_switches_type_and_metadata():
    exp = _make_mnist1m(max_train_samples=512)
    assert exp.experiment_type == "mnist1m_sampled_classification"
    assert exp.get_sweep_metadata(init_key=11) == {"subsample_seed": 11}
    assert exp.to_params_dict()["max_train_samples"] == 512

    train_ds = {
        "image": np.zeros((1024, 28, 28, 1), dtype=np.float32),
        "label": np.zeros((1024,), dtype=np.int32),
    }
    assert not exp.should_skip_batch_size(batch_size=512, train_ds=train_ds)
    assert exp.should_skip_batch_size(batch_size=1024, train_ds=train_ds)


def test_sampled_alias_remains_supported():
    sampled_alias = MNIST1MSampledExperiment(
        N=32,
        L=2,
        parameterization=Parameterization.MUP,
        gamma=1.0,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
        num_epochs=2,
        max_train_samples=1024,
    )
    assert isinstance(sampled_alias, MNIST1MExperiment)
    assert sampled_alias.experiment_type == "mnist1m_sampled_classification"


def test_sampled_specs_use_unified_class():
    sampled_specs = [spec for spec in list_main_experiment_specs() if spec.family == "mnist1m_sampled_classification"]
    assert sampled_specs
    assert all(spec.experiment_cls is MNIST1MExperiment for spec in sampled_specs)
    built = sampled_specs[0].build()
    assert built.experiment_type == "mnist1m_sampled_classification"
    assert built.max_train_samples is not None
