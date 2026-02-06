"""
Compatibility shim for experiment registration.

The registry is now organized under ``batch_size_studies.studies``. This module
keeps legacy imports working by re-exporting the prior builder symbols and
triggering registration side effects.
"""

from .studies.catalog import *  # noqa: F401,F403
from .studies.linear_teacher_catalog import linear_teacher_specs
from .studies.mnist1m_catalog import MNIST_GAMMA_SWEEP, mnist1m_sampled_specs, mnist1m_specs

__all__ = [
    "MNIST_GAMMA_SWEEP",
    "linear_teacher_specs",
    "mnist1m_specs",
    "mnist1m_sampled_specs",
]
