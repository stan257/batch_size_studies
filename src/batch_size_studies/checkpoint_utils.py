"""Compatibility alias for checkpoint internals now hosted under `batch_size_studies.engine`."""

import sys as _sys

from .engine import checkpoint_utils as _checkpoint_utils

_sys.modules[__name__] = _checkpoint_utils
