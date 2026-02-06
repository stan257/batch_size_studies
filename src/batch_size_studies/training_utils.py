"""Compatibility alias for training utilities now hosted under `batch_size_studies.engine`."""

import sys as _sys

from .engine import training_utils as _training_utils

_sys.modules[__name__] = _training_utils
