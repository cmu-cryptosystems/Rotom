"""
Frontend modules for the Rotom FHE compiler.

This package provides the frontend interfaces for defining tensor computations
that will be compiled to homomorphic encryption operations.

Modules:
    tensor: High-level tensor operations and expressions for FHE computations
    rotom_pytorch: PyTorch-like interface for familiar API with HE support
"""

from .tensor import TensorOp, TensorTerm
from .rotom_pytorch import Tensor, torch, nn, optim

__all__ = ["TensorTerm", "TensorOp", "Tensor", "torch", "nn", "optim"]
