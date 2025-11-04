from dataclasses import InitVar, dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from batch_size_studies.checkpoint_utils import CheckpointManager
from batch_size_studies.data_iterators import EpochBasedDataIterator
from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiments import MNISTExperiment, SyntheticExperimentFixedTime
from batch_size_studies.hessian_evaluator import HessianEvaluator


@dataclass(frozen=True)
class DummyMNISTExperiment(MNISTExperiment):
    train_dataset: InitVar[dict | None] = None

    def __post_init__(self, train_dataset):
        super().__post_init__()
        if train_dataset is None:
            raise ValueError("train_dataset must be provided for DummyMNISTExperiment.")
        object.__setattr__(self, "_train_dataset", train_dataset)

    def prepare_datasets(self, init_key: int, **kwargs):
        return self._train_dataset, None


def _initialize_params(experiment, directory: str, init_key: int = 0):
    checkpoint_manager = CheckpointManager(experiment, directory=directory)
    model_instance = experiment.create_model_instance()
    widths = experiment.get_model_widths()
    checkpoint_manager.initialize_and_save_initial_params(init_key, model_instance, widths)


def test_hessian_evaluator_replays_epoch_iterator_order(tmp_path):
    train_images = jnp.arange(6 * 784, dtype=jnp.float32).reshape(6, 784)
    train_labels = jnp.arange(6, dtype=jnp.int32)
    train_ds = {"image": train_images, "label": train_labels}

    experiment = DummyMNISTExperiment(
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
        N=4,
        L=2,
        parameterization=Parameterization.SP,
        gamma=1.0,
        num_epochs=1,
        train_dataset=train_ds,
    )
    directory = tmp_path / "experiments"
    _initialize_params(experiment, str(directory), init_key=0)

    run_key = RunKey(batch_size=2, eta=0.1)
    evaluator = HessianEvaluator(
        experiment=experiment,
        run_key=run_key,
        step=None,
        directory=str(directory),
        num_hessian_samples=6,
        hessian_batch_size=2,
        init_key=0,
    )

    loader_inputs = jnp.concatenate([batch[0] for batch in evaluator.data_loader], axis=0)
    loader_labels = jnp.concatenate([batch[1] for batch in evaluator.data_loader], axis=0)

    iterator = EpochBasedDataIterator(
        train_ds=train_ds,
        batch_size=run_key.batch_size,
        num_epochs=experiment.num_epochs,
        init_key=0,
    )
    expected_inputs = []
    expected_labels = []
    for batch_inputs, batch_labels in iterator:
        expected_inputs.append(batch_inputs)
        expected_labels.append(batch_labels)
    expected_inputs = jnp.concatenate(expected_inputs, axis=0)[:6]
    expected_labels = jnp.concatenate(expected_labels, axis=0)[:6]

    np.testing.assert_array_equal(np.array(loader_inputs), np.array(expected_inputs))
    np.testing.assert_array_equal(np.array(loader_labels), np.array(expected_labels))


def test_hessian_evaluator_generates_online_samples(tmp_path):
    experiment = SyntheticExperimentFixedTime(
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
        N=4,
        L=2,
        parameterization=Parameterization.SP,
        gamma=1.0,
        D=3,
        P=64,
        K=2,
        num_steps=10,
    )
    directory = tmp_path / "experiments"
    _initialize_params(experiment, str(directory), init_key=0)

    run_key = RunKey(batch_size=8, eta=0.1)
    evaluator = HessianEvaluator(
        experiment=experiment,
        run_key=run_key,
        step=None,
        directory=str(directory),
        num_hessian_samples=32,
        hessian_batch_size=8,
        init_key=0,
    )

    total_samples = sum(batch[0].shape[0] for batch in evaluator.data_loader)
    assert total_samples > 0
    assert all(batch[0].shape[0] == batch[1].shape[0] for batch in evaluator.data_loader)


def test_hessian_evaluator_keeps_partial_batches(tmp_path):
    train_images = jnp.arange(3 * 784, dtype=jnp.float32).reshape(3, 784)
    train_labels = jnp.array([0, 1, 2])
    train_ds = {"image": train_images, "label": train_labels}

    experiment = DummyMNISTExperiment(
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
        N=4,
        L=2,
        parameterization=Parameterization.SP,
        gamma=1.0,
        num_epochs=1,
        train_dataset=train_ds,
    )
    directory = tmp_path / "experiments"
    _initialize_params(experiment, str(directory), init_key=0)

    evaluator = HessianEvaluator(
        experiment=experiment,
        run_key=None,
        step=None,
        directory=str(directory),
        num_hessian_samples=10,
        hessian_batch_size=5,
        init_key=0,
    )

    batch_sizes = [batch[0].shape[0] for batch in evaluator.data_loader]
    assert batch_sizes[-1] == 3
