"""Compatibility alias for trainer internals now hosted under `batch_size_studies.engine`."""

import sys as _sys

from .engine import trainer as _trainer

_sys.modules[__name__] = _trainer
