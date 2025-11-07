"""Experiment specification registry used for configuring sweeps."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .definitions import LossType, OptimizerType
from .experiments import ExperimentBase


@dataclass(frozen=True)
class ExperimentSpec:
    """Declarative description of an experiment configuration."""

    name: str
    experiment_cls: type[ExperimentBase]
    kwargs: Dict[str, Any]
    family: str
    optimizer: OptimizerType | None = None
    loss_type: LossType | None = None

    def build(self) -> ExperimentBase:
        return self.experiment_cls(**self.kwargs)

    def matches(
        self,
        experiment_types: Sequence[str] | None = None,
        optimizer: OptimizerType | None = None,
        loss_type: LossType | None = None,
    ) -> bool:
        if experiment_types is not None and self.family not in experiment_types:
            return False
        target_optimizer = self.optimizer or self.kwargs.get("optimizer")
        if optimizer is not None and target_optimizer != optimizer:
            return False
        target_loss = self.loss_type or self.kwargs.get("loss_type")
        if loss_type is not None and target_loss != loss_type:
            return False
        return True


_EXPERIMENT_SPEC_REGISTRY: List[ExperimentSpec] = []


def register_experiment_specs(*specs: ExperimentSpec) -> Tuple[ExperimentSpec, ...]:
    """Adds one or more specs to the global registry."""

    _EXPERIMENT_SPEC_REGISTRY.extend(specs)
    return specs


def register_spec_builder(builder_fn):
    """Decorator that registers all specs returned by the builder function."""

    specs = builder_fn()
    if not isinstance(specs, Iterable):
        raise TypeError("Spec builder must return an iterable of ExperimentSpec instances.")
    register_experiment_specs(*tuple(specs))
    return builder_fn


def list_registered_specs() -> Tuple[ExperimentSpec, ...]:
    return tuple(_EXPERIMENT_SPEC_REGISTRY)


def clear_registry():
    """Utility for tests that need a clean registry."""

    _EXPERIMENT_SPEC_REGISTRY.clear()
