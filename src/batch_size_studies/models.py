from dataclasses import dataclass, field
from typing import Callable, List

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import relu

from .definitions import Parameterization
from .protocols import ModelProtocol


@dataclass(frozen=True)
class MLP(ModelProtocol):
    """
    A dataclass to define and manage a Multi-Layer Perceptron (MLP).

    This class assumes all hidden layers have the same width.

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
        keys = jr.split(jr.PRNGKey(init_key), depth)

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
class LinearModel(ModelProtocol):
    """
    A simple linear model: y = x @ W, no bias.
    """

    def init_params(self, init_key: int, widths: List[int], **kwargs) -> jnp.ndarray:
        if len(widths) != 2:
            raise ValueError(
                f"LinearModel expects `widths` to be a list of length 2 ([input_dim, output_dim]), "
                f"but got {len(widths)}."
            )
        input_dim, output_dim = widths
        # return single array since no bias
        W = jnp.zeros((input_dim, output_dim))
        return W

    def __call__(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for the linear model."""
        W = params
        return jnp.dot(x, W)


class CenteredModel:
    """
    A wrapper for a JAX model to compute centered outputs. Does
        L(p) = loss(model(p) - model(p0)),
    where p0 are the initial parameters.
    """

    def __init__(self, model, params0):
        self.model = model
        self.params0 = params0
        # Jit the entire centered computation for efficiency
        self._centered_fn = jax.jit(self._compute_centered)

    def _compute_centered(self, params, inputs):
        return self.model(params, inputs) - self.model(self.params0, inputs)

    def __call__(self, params, inputs):
        return self._centered_fn(params, inputs)
