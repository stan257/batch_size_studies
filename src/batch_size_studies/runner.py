"""Compatibility alias for runner internals now hosted under `batch_size_studies.engine`."""

import sys as _sys

from .engine import runner as _runner

_sys.modules[__name__] = _runner
