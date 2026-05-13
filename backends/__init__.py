"""
Backend implementations for the FHE Compiler.

This module provides various backend implementations for executing FHE circuits:
- Toy: Plaintext simulation for development and testing
- CKKS: Production CKKS implementation using OpenFHE
- HEIR: Research framework integration for MLIR

Each backend implements a common interface for executing FHE circuits
with different underlying homomorphic encryption libraries.
"""

from .heir.heir import HEIR
from .toy import Toy

__all__ = ["Toy", "CKKS", "HEIR"]


def __getattr__(name):
    if name == "CKKS":
        from .openfhe_backend import CKKS

        return CKKS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
