import logging

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from .checkpoint_utils import CheckpointManager, load_experiment_weights
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
        """
        self.experiment = experiment
        self.run_key = run_key
        self.step = step
        self.directory = directory
        self.key = jr.PRNGKey(42)  # A fixed key for reproducibility

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
        # For sampled experiments, ensure we use the same data subset as training.
        cm = CheckpointManager(self.experiment, directory=self.directory)
        metadata = cm.load_sweep_metadata()
        subsample_seed = metadata.get("subsample_seed")
        if subsample_seed is not None:
            logging.info(f"Found and using subsample_seed: {subsample_seed}")

        # Use the experiment's own method to prepare the dataset.
        # The Hessian should be evaluated on the training distribution.
        train_ds, _ = self.experiment.prepare_datasets(init_key=self.key.sum(), forced_subsample_seed=subsample_seed)
        if train_ds is None:
            raise ValueError("Hessian evaluation requires a dataset, but prepare_datasets returned None.")
        self.data_loader = self._create_data_loader(train_ds, num_hessian_samples, hessian_batch_size)

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

    def _create_data_loader(self, dataset, num_samples, batch_size):
        """Creates a data loader from a dataset, subsampling if necessary."""
        if isinstance(dataset, dict):  # MNIST-like
            images, labels = dataset["image"], dataset["label"]
        elif isinstance(dataset, tuple):  # Synthetic-like
            images, labels = dataset[0], dataset[1]
        else:
            raise TypeError(f"Unsupported dataset type for Hessian evaluation: {type(dataset)}")

        images, labels = images[:num_samples], labels[:num_samples]
        if images.ndim > 2:
            images = images.reshape(images.shape[0], -1)

        data_loader = []
        for i in range(0, len(images), batch_size):
            batch_img, batch_lbl = images[i : i + batch_size], labels[i : i + batch_size]
            if len(batch_img) == batch_size:
                data_loader.append((batch_img, batch_lbl))

        if not data_loader:
            raise ValueError("Data loader is empty. Check num_samples and batch_size.")
        return data_loader

    def _get_outer_loss_fn(self):
        def loss_fn_outer(model_output, targets):
            if self.experiment.loss_type == LossType.XENT:
                return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=model_output, labels=targets))
            elif self.experiment.loss_type == LossType.MSE:
                if isinstance(self.experiment, (MNISTExperiment, MNIST1MExperiment, MNIST1MSampledExperiment)):
                    targets_one_hot = jax.nn.one_hot(targets, num_classes=self.experiment.num_outputs)
                    return jnp.mean((model_output - targets_one_hot) ** 2)
                else:  # Regression
                    return jnp.mean((model_output - targets) ** 2)
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
