from dataclasses import dataclass, field
from typing import Callable, List

import jax.numpy as jnp
import jax.random as jr
from jax.nn import relu

from .definitions import Parameterization


@dataclass(frozen=True)
class MLP:
    """
    A dataclass to define and manage a Multi-Layer Perceptron (MLP).

    This class encapsulates the model's architecture, scaling, and parameter
    initialization logic. We assume all hidden layers have the same width.

    Attributes:
        parameterization (Parameterization): The parameterization scale, either
                                             Parameterization.SP or Parameterization.MUP.
        gamma (float): The richness parameter, used for scaling the output layer.
    """

    parameterization: Parameterization
    gamma: float = 1.0

    # This is a private attribute to hold the correct forward pass function
    _apply_fn: Callable = field(init=False, repr=False)

    def __post_init__(self):
        """
        Selects the correct forward pass function after the instance is created.
        """
        if not isinstance(self.parameterization, Parameterization):
            raise TypeError(
                f"parameterization must be a member of the Parameterization enum, "
                f"but got type {type(self.parameterization).__name__}."
            )
        match self.parameterization:
            case Parameterization.SP:
                forward_pass = self._mlp_forward_pass_sp
            case Parameterization.MUP:
                forward_pass = self._mlp_forward_pass_mup

        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "_apply_fn", forward_pass)

    def init_params(self, init_key: int, widths: List[int], sigma_w: float = 1.0) -> List[jnp.ndarray]:
        """
        Initializes the parameters (weights) for the MLP.

        Args:
            init_key (int): A single integer seed to generate all JAX random keys.
            widths (List[int]): A list of layer widths, e.g., [input_dim, hidden_1, ..., output_dim].
            sigma_w (float, optional): The standard deviation of the initial weights. Defaults to 1.0.

        Returns:
            List[jnp.ndarray]: A list of weight matrices for the MLP.
        """
        depth = len(widths) - 1
        keys = jr.split(jr.key(init_key), depth)

        params = []
        for i in range(len(widths) - 1):
            # These are initialized with mean 0 and variance sigma_w^2
            params.append(jr.normal(keys[i], (widths[i], widths[i + 1])) * sigma_w)
        return params

    def _mlp_forward_pass_sp(self, params: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        """The forward pass logic for Standard Parameterization (SP)."""
        for i in range(len(params) - 1):
            x = jnp.dot(x, params[i]) / jnp.sqrt(x.shape[-1])
            x = relu(x)

        # Output layer scaling with sqrt(width)
        x = jnp.dot(x, params[-1]) / (self.gamma * jnp.sqrt(x.shape[-1]))
        return x

    def _mlp_forward_pass_mup(self, params: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        """The forward pass logic for Maximal Update Parameterization (muP)."""
        for i in range(len(params) - 1):
            x = jnp.dot(x, params[i]) / jnp.sqrt(x.shape[-1])
            x = relu(x)

        # Output layer scaling with width
        x = jnp.dot(x, params[-1]) / (self.gamma * x.shape[-1])
        return x

    def __call__(self, params: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        """
        Makes the MLP instance callable, executing the forward pass.

        This allows you to treat an instance of this class like a function:

        Example:
            model = MLP(parametrization=Parameterization.SP)
            output = model(params, input_data)
        """
        return self._apply_fn(params, x)
