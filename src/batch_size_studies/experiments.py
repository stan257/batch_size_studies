"""
Legacy shim that re-exports experiment dataclasses.

New code should import from :mod:`batch_size_studies.experiment_types`, but this
module remains for backwards compatibility with older imports/pickles.
"""

from .data_loading import load_datasets, load_mnist1m_dataset  # noqa: F401
from .experiment_types import *  # noqa: F401,F403
from .experiment_types import __all__ as _EXPERIMENT_TYPES_ALL
from .experiment_types.mnist import _load_mnist_dataset, _subsample_mnist_data  # noqa: F401

__all__ = _EXPERIMENT_TYPES_ALL
