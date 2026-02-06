"""Compatibility alias for protocols now hosted under `batch_size_studies.engine`."""

import sys as _sys

from .engine import protocols as _protocols

_sys.modules[__name__] = _protocols
