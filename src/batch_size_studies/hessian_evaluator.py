import logging

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from .checkpoint_utils import CheckpointManager, load_experiment_weights
from .data_iterators import EpochBasedDataIterator, OnlineDataIterator
from .definitions import LossType, RunKey
from .experiments import (
    ExperimentBase,
    LinearStudentExperiment,
    MLPStudentExperiment,
    MNIST1MExperiment,
    MNIST1MSampledExperiment,
    MNISTExperiment,
)
from .hessian import JaxHessian
from .models import MLP, CenteredModel, LinearModel
from .paths import EXPERIMENTS_DIR


class HessianEvaluator:
    """
    A class to evaluate Hessian properties (eigenvalues, trace, density) for a
    given experiment at a specific point in training (or at initialization).
    """

    def __init__(
        self,
        experiment: ExperimentBase,
        run_key: RunKey | None = None,
        step: int | None = None,
        directory: str = EXPERIMENTS_DIR,
        num_hessian_samples: int = 1024,
        hessian_batch_size: int = 128,
        init_key: int | None = None,
    ):
        """
        Initializes the HessianEvaluator.

        Args:
            experiment: The experiment configuration object.
            run_key: The specific run (batch_size, eta) to load weights from.
                     If None, evaluation is at initialization.
            step: The training step to load weights from. Required if run_key is not None.
            directory: The base directory for experiments.
            num_hessian_samples: Number of data samples to use for Hessian estimation.
            hessian_batch_size: Batch size for Hessian-vector products.
            init_key: Optional override for the sweep init key. If None, the value saved
                      alongside the sweep is used (falling back to 0).
        """
        self.experiment = experiment
        self.run_key = run_key
        self.step = step
        self.directory = directory
        self.key = jr.PRNGKey(42)  # A fixed key for reproducibility
        self.num_hessian_samples = num_hessian_samples
        self.hessian_batch_size = hessian_batch_size

        # --- 1. Load Parameters ---
        self.params0 = self._load_initial_params()
        if self.params0 is None:
            raise FileNotFoundError("Could not load initial parameters (params0). Run experiment first.")

        if run_key and step is not None:
            self.params = self._load_trained_params()
            if self.params is None:
                raise FileNotFoundError(f"Could not load parameters for {run_key} at step {step}.")
            logging.info(f"Evaluating Hessian for {run_key} at step {step}.")
        else:
            self.params = self.params0
            logging.info("Evaluating Hessian at initialization (params0).")

        # --- 2. Load Data ---
        cm = CheckpointManager(self.experiment, directory=self.directory)
        metadata = cm.load_sweep_metadata()
        subsample_seed = metadata.get("subsample_seed")
        if subsample_seed is not None:
            logging.info(f"Found and using subsample_seed: {subsample_seed}")
        stored_init_key = metadata.get("init_key")
        if init_key is None:
            self.init_key = int(stored_init_key) if stored_init_key is not None else 0
        else:
            self.init_key = init_key
        if stored_init_key is not None and init_key is not None and stored_init_key != init_key:
            logging.info(
                "Overriding stored init_key=%s with provided init_key=%s for Hessian evaluation.",
                stored_init_key,
                init_key,
            )

        train_ds, _ = self.experiment.prepare_datasets(
            init_key=self.init_key,
            forced_subsample_seed=subsample_seed,
        )
        inputs, targets = self._collect_training_samples(train_ds, num_hessian_samples)
        self.data_loader = self._batch_samples(inputs, targets, hessian_batch_size)

        # --- 3. Instantiate Model and Loss ---
        if isinstance(self.experiment, LinearStudentExperiment):
            model_instance = LinearModel()
            # For linear models, we don't center the output.
            model_to_use = model_instance
        elif isinstance(self.experiment, MLPStudentExperiment):
            model_instance = MLP(self.experiment.parameterization, self.experiment.gamma)
            # For MLP models, we use the centered output to match training loss.
            model_to_use = CenteredModel(model_instance, self.params0)
        else:
            raise TypeError(f"Unknown student model for experiment type: {type(self.experiment).__name__}")

        loss_fn_outer = self._get_outer_loss_fn()

        # --- 4. Instantiate JaxHessian ---
        self.hessian_computer = JaxHessian(model=model_to_use, loss_fn=loss_fn_outer, data_loader=self.data_loader)
        logging.info("HessianEvaluator initialized successfully.")

    def _load_initial_params(self):
        checkpoint_manager = CheckpointManager(self.experiment, directory=self.directory)
        return checkpoint_manager.load_initial_params()

    def _load_trained_params(self):
        return load_experiment_weights(
            self.experiment,
            batch_size=self.run_key.batch_size,
            eta=self.run_key.eta,
            directory=self.directory,
            step_to_load=self.step,
        )

    def _collect_training_samples(self, train_ds, num_samples: int):
        if train_ds is None:
            return self._collect_online_samples(num_samples)
        return self._collect_offline_samples(train_ds, num_samples)

    def _collect_offline_samples(self, dataset, num_samples: int):
        if self.run_key is not None:
            samples = self._samples_via_epoch_iterator(dataset, num_samples)
            if samples is not None:
                return samples

        images, labels = self._dataset_to_arrays(dataset)
        dataset_size = images.shape[0]
        if dataset_size == 0:
            raise ValueError("Training dataset is empty; cannot evaluate Hessian.")

        max_take = min(num_samples, dataset_size)
        shuffle_key = jr.PRNGKey(self.init_key + 7)
        permutation = jr.permutation(shuffle_key, dataset_size)
        selection = permutation[:max_take]

        return images[selection], labels[selection]

    def _samples_via_epoch_iterator(self, dataset, num_samples: int):
        dataset_size = self._get_dataset_size(dataset)
        if dataset_size == 0:
            return None

        batch_size = min(self.run_key.batch_size, dataset_size)
        if batch_size <= 0:
            return None

        steps_per_epoch = dataset_size // batch_size
        if steps_per_epoch == 0:
            return None

        num_epochs = getattr(self.experiment, "num_epochs", 1)
        iterator = EpochBasedDataIterator(
            train_ds=dataset,
            batch_size=batch_size,
            num_epochs=max(1, num_epochs),
            init_key=self.init_key,
        )
        gathered = self._gather_from_iterator(iterator, num_samples)
        return gathered

    def _collect_online_samples(self, num_samples: int):
        if not hasattr(self.experiment, "generate_data"):
            raise ValueError(
                "Experiment does not provide generate_data; provide a dataset or override init_key for offline use."
            )

        batch_size = self.run_key.batch_size if self.run_key is not None else self.hessian_batch_size
        online_iterator = OnlineDataIterator(
            experiment=self.experiment,
            batch_size=max(1, batch_size),
            start_step=0,
            initial_batch_key_seed=0,
        )
        gathered = self._gather_from_iterator(online_iterator, num_samples)
        if gathered is not None:
            return gathered

        # Fallback: manually accumulate synthetic data using deterministic seeds.
        inputs_list = []
        targets_list = []
        total = 0
        seed = 0
        while total < num_samples:
            data_key = jr.key(self.init_key + seed)
            inputs, targets = self.experiment.generate_data(data_key)
            inputs = jnp.asarray(inputs)
            targets = jnp.asarray(targets)
            take = min(inputs.shape[0], num_samples - total)
            if take == 0:
                break
            inputs_list.append(inputs[:take])
            targets_list.append(targets[:take])
            total += take
            seed += 1

        if not inputs_list:
            raise ValueError("Unable to generate data for online experiment; dataset size may be zero.")

        inputs = jnp.concatenate(inputs_list, axis=0)
        targets = jnp.concatenate(targets_list, axis=0)
        return inputs, targets

    def _dataset_to_arrays(self, dataset):
        if isinstance(dataset, dict):
            images = jnp.asarray(dataset["image"])
            labels = jnp.asarray(dataset["label"])
        elif isinstance(dataset, tuple):
            images = jnp.asarray(dataset[0])
            labels = jnp.asarray(dataset[1])
        else:
            raise TypeError(f"Unsupported dataset type for Hessian evaluation: {type(dataset)}")

        if images.ndim > 2:
            images = images.reshape(images.shape[0], -1)

        return images, labels

    def _get_dataset_size(self, dataset) -> int:
        if isinstance(dataset, dict):
            return int(dataset["image"].shape[0])
        if isinstance(dataset, tuple):
            return int(dataset[0].shape[0])
        return 0

    def _gather_from_iterator(self, iterator, num_samples: int):
        inputs_batches = []
        targets_batches = []
        collected = 0

        for batch_inputs, batch_targets in iterator:
            batch_inputs = jnp.asarray(batch_inputs)
            batch_targets = jnp.asarray(batch_targets)
            inputs_batches.append(batch_inputs)
            targets_batches.append(batch_targets)
            collected += batch_inputs.shape[0]
            if collected >= num_samples:
                break

        if collected == 0:
            return None

        inputs = jnp.concatenate(inputs_batches, axis=0)[: min(collected, num_samples)]
        targets = jnp.concatenate(targets_batches, axis=0)[: min(collected, num_samples)]
        return inputs, targets

    def _batch_samples(self, inputs: jnp.ndarray, targets: jnp.ndarray, batch_size: int):
        if inputs.shape[0] == 0:
            raise ValueError("Collected zero samples for Hessian evaluation.")

        data_loader = []
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            data_loader.append((inputs[start:end], targets[start:end]))
        return data_loader

    def _get_outer_loss_fn(self):
        def loss_fn_outer(model_output, targets):
            if self.experiment.loss_type == LossType.XENT:
                return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=model_output, labels=targets))
            elif self.experiment.loss_type == LossType.MSE:
                if isinstance(self.experiment, (MNISTExperiment, MNIST1MExperiment, MNIST1MSampledExperiment)):
                    targets_one_hot = jax.nn.one_hot(targets, num_classes=self.experiment.num_outputs)
                    diff = model_output - targets_one_hot
                else:  # Regression
                    diff = model_output - targets
                diff = diff.reshape(diff.shape[0], -1)
                return 0.5 * jnp.mean(jnp.sum(diff**2, axis=1))
            raise NotImplementedError(f"Loss type {self.experiment.loss_type} not supported.")

        return loss_fn_outer

    def top_eigenvalues(self, top_n=1, max_iter=100, tol=1e-3):
        """Computes the top N eigenvalues of the Hessian."""
        logging.info(f"Computing top {top_n} eigenvalue(s)...")
        key, subkey = jr.split(self.key)
        self.key = key  # Update state for next call

        eigenvalues, eigenvectors = self.hessian_computer.eigenvalues(
            params=self.params, key=subkey, top_n=top_n, max_iter=max_iter, tol=tol
        )
        return eigenvalues, eigenvectors

    def trace(self, max_iter=100):
        """Computes the trace of the Hessian using Hutchinson's method."""
        logging.info("Computing Hessian trace...")
        key, subkey = jr.split(self.key)
        self.key = key

        trace_val, _ = self.hessian_computer.trace(params=self.params, key=subkey, max_iter=max_iter)
        return trace_val

    def density(self, num_iterations=100, num_runs=10):
        """Computes the eigenvalue density of the Hessian using Stochastic Lanczos Quadrature."""
        logging.info("Computing Hessian eigenvalue density...")
        key, subkey = jr.split(self.key)
        self.key = key

        eigen_list, weight_list = self.hessian_computer.density(
            params=self.params, key=subkey, num_iterations=num_iterations, num_runs=num_runs
        )
        return eigen_list, weight_list
