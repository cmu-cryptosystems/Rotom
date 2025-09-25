"""
Layout Intermediate Representations (IR) used in Rotom.

This module provides the core intermediate representations used 
throughout Rotom. 

Key Components:

- Dim: Represents tensor dimensions with extent, stride, and type
- Roll: Handles rotation operations and permutations
- Layout IR: Defines how tensor data is packed into HE vectors
- Kernel IR : Represents tensor operations using layout IR
- HE IR: Represents homomorphic encryption terms and operations
- Analysis: Shape and secret analysis passes
"""

from .dim import Dim, DimType
from .layout import Layout
from .roll import Roll
from .kernel import Kernel, KernelOp
from .he import HETerm, HEOp
from .kernel_cost import KernelCost

__all__ = [
    'Dim', 'DimType', 'Layout', 'Roll', 'KernelOp', 'Kernel', 'HEOp', 'HETerm'
]
