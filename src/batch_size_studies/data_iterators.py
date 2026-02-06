"""Compatibility alias for data iterators now hosted under `batch_size_studies.engine`."""

import sys as _sys

from .engine import data_iterators as _data_iterators

_sys.modules[__name__] = _data_iterators
