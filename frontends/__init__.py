"""
Frontend modules for the Rotom FHE compiler.

This package provides the frontend interfaces for defining tensor computations
that will be compiled to homomorphic encryption operations.

Modules:
    tensor: High-level tensor operations and expressions for FHE computations
"""

from .tensor import TensorOp, TensorTerm

__all__ = ["TensorTerm", "TensorOp"]
