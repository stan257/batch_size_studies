import batch_size_studies.checkpoint_utils as legacy_checkpoint_utils
import batch_size_studies.data_iterators as legacy_data_iterators
import batch_size_studies.runner as legacy_runner
import batch_size_studies.trainer as legacy_trainer
from batch_size_studies.engine import checkpoint_utils as engine_checkpoint_utils
from batch_size_studies.engine import data_iterators as engine_data_iterators
from batch_size_studies.engine import runner as engine_runner
from batch_size_studies.engine import trainer as engine_trainer


def test_runner_shim_reexports_entrypoints():
    assert legacy_runner.run_experiment_sweep is engine_runner.run_experiment_sweep
    assert legacy_runner.TrialContext is engine_runner.TrialContext


def test_trainer_shim_reexports_core_runners():
    assert legacy_trainer.MNISTTrialRunner is engine_trainer.MNISTTrialRunner
    assert legacy_trainer.SyntheticFixedDataTrialRunner is engine_trainer.SyntheticFixedDataTrialRunner


def test_checkpoint_and_iterator_shims_reexport_types():
    assert legacy_checkpoint_utils.CheckpointManager is engine_checkpoint_utils.CheckpointManager
    assert legacy_data_iterators.EpochBasedDataIterator is engine_data_iterators.EpochBasedDataIterator
