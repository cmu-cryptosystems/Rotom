"""
Backend implementations for the FHE Compiler.

This module provides various backend implementations for executing FHE circuits:
- Toy: Plaintext simulation for development and testing
- CKKS: Production CKKS implementation using OpenFHE
- HEIR: Research framework integration for MLIR

Each backend implements a common interface for executing FHE circuits
with different underlying homomorphic encryption libraries.
"""

from .toy import Toy
from .openfhe_backend import CKKS

__all__ = ['Toy', 'CKKS', 'HEIR']
