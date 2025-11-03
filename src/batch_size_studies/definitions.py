from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Protocol

import jax.numpy as jnp


class ModelProtocol(Protocol):
    def init_params(self, init_key: int, widths: List[int]) -> Any: ...

    def __call__(self, params: Any, inputs: jnp.ndarray) -> jnp.ndarray: ...


class Parameterization(Enum):
    """Defines the valid parameterization scales for the MLP model."""

    SP = "SP"
    MUP = "muP"


class LossType(Enum):
    """Defines the loss function type."""

    MSE = "MSE"
    XENT = "XENT"  # Cross-Entropy


class OptimizerType(Enum):
    """Defines the optimizer type."""

    SGD = "SGD"
    ADAM = "Adam"


@dataclass(frozen=True)
class RunKey:
    batch_size: int
    eta: float

    def __post_init__(self):
        if not isinstance(self.batch_size, int):
            raise TypeError(f"RunKey 'batch_size' must be an integer, but got type {type(self.batch_size).__name__}.")
        if not isinstance(self.eta, float):
            raise TypeError(f"RunKey 'eta' must be a float, but got type {type(self.eta).__name__}.")

    @property
    def temp(self):
        return self.eta / self.batch_size
