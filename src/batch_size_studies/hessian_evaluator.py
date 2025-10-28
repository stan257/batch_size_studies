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
from .models import MLP, LinearModel
from .paths import EXPERIMENTS_DIR


class CenteredModel:
    """
    A wrapper for a JAX model to compute centered outputs.

    This is used to match the loss function structure from training, which is
    L(p) = loss(model(p) - model(p0)), where p0 are the initial parameters.
    The JaxHessian class will compute the Hessian of loss(model(p)), so by
    passing this wrapper as the model, we ensure it computes the Hessian of
    the correct training loss.
    """

    def __init__(self, model, params0):
        self.model = model
        self.params0 = params0
        # The model's __call__ needs to be jitted for performance inside HVP loop
        self.apply_fn = jax.jit(self.model)

    def __call__(self, params, inputs):
        """
        Computes model(params, inputs) - model(params0, inputs).
        """
        return self.apply_fn(params, inputs) - self.apply_fn(self.params0, inputs)


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
        data_loader = self._prepare_data_loader(num_hessian_samples, hessian_batch_size)

        # --- 3. Instantiate Model and Loss ---
        if isinstance(self.experiment, LinearStudentExperiment):
            model_instance = LinearModel()
        elif isinstance(self.experiment, MLPStudentExperiment):
            model_instance = MLP(self.experiment.parameterization, self.experiment.gamma)
        else:
            raise TypeError(f"Unknown student model for experiment type: {type(self.experiment).__name__}")

        centered_model = CenteredModel(model_instance, self.params0)
        loss_fn_outer = self._get_outer_loss_fn()

        # --- 4. Instantiate JaxHessian ---
        self.hessian_computer = JaxHessian(model=centered_model, loss_fn=loss_fn_outer, data_loader=data_loader)
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

    def _prepare_data_loader(self, num_samples, batch_size):
        data_key = jr.PRNGKey(getattr(self.experiment, "seed", 42))

        if isinstance(self.experiment, (MNISTExperiment, MNIST1MExperiment, MNIST1MSampledExperiment)):
            if isinstance(self.experiment, (MNIST1MExperiment, MNIST1MSampledExperiment)):
                from .data_loading import load_mnist1m_dataset

                dataset_loader_fn = load_mnist1m_dataset
            else:
                from .data_loading import load_datasets

                dataset_loader_fn = load_datasets
            (_, _), (images, labels) = dataset_loader_fn()
        elif hasattr(self.experiment, "generate_data"):  # For all synthetic experiments
            images, labels = self.experiment.generate_data(data_key)
        else:
            raise TypeError(f"Cannot prepare data for experiment type {type(self.experiment)}")

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
