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
        depth = len(widths) - 1
        keys = jr.split(jr.key(init_key), depth)

        params = []
        for i in range(len(widths) - 1):
            params.append(jr.normal(keys[i], (widths[i], widths[i + 1])) * sigma_w)
        return params

    def _mlp_forward_pass_sp(self, params: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        for i in range(len(params) - 1):
            x = jnp.dot(x, params[i]) / jnp.sqrt(x.shape[-1])
            x = relu(x)

        x = jnp.dot(x, params[-1]) / (self.gamma * jnp.sqrt(x.shape[-1]))
        return x

    def _mlp_forward_pass_mup(self, params: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        for i in range(len(params) - 1):
            x = jnp.dot(x, params[i]) / jnp.sqrt(x.shape[-1])
            x = relu(x)

        x = jnp.dot(x, params[-1]) / (self.gamma * x.shape[-1])
        return x

    def __call__(self, params: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        return self._apply_fn(params, x)


@dataclass(frozen=True)
class LinearModel:
    """
    A simple linear model: y = x @ W, no bias.
    """

    def init_params(self, init_key: int, widths: List[int], **kwargs) -> List[jnp.ndarray]:
        """
        Initializes parameters with zeros.

        Args:
            init_key: A random key (ignored, as initialization is deterministic).
            widths: A list of [input_dim, output_dim].
            **kwargs: Ignored, for API compatibility with MLP.

        Returns:
            A list containing a zero-initialized weight matrix and bias vector.
        """
        if len(widths) != 2:
            raise ValueError(
                f"LinearModel expects `widths` to be a list of length 2 ([input_dim, output_dim]), "
                f"but got {len(widths)}."
            )
        input_dim, output_dim = widths
        # A linear model is like a single-layer network with weights and bias.
        # We return a list to be consistent with the MLP parameter structure.
        W = jnp.zeros((input_dim, output_dim))
        return W

    def __call__(self, params: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for the linear model."""
        W = params
        return jnp.dot(x, W)
