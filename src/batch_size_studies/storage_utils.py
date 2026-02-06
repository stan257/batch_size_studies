"""Compatibility alias for storage helpers now hosted under `batch_size_studies.engine`."""

import sys as _sys

from .engine import storage_utils as _storage_utils

_sys.modules[__name__] = _storage_utils
