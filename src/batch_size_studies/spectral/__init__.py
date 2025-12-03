"""
Spectral analysis helpers (HVP-based Hessian utilities, evaluators, caching).
"""

from . import spectral_utils
from .hessian import JaxHessian
from .hessian_evaluator import HessianEvaluator

__all__ = ["JaxHessian", "HessianEvaluator", "spectral_utils"]
