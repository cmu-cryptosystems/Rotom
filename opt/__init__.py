"""
High-level optimization passes for the FHE Compiler.

This module provides optimization passes that operate on kernels and layouts
to improve the efficiency of homomorphic encryption computations. These
optimizations focus on reducing the cost of expensive operations like
rotations and multiplications.

Key Optimization Passes:
- Roll Propagation: Moves rolls to optimal positions
- Roll Reordering: Reorders roll operations for efficiency
- BSGS Matrix Multiplication: Reduces rotation costs in matrix multiplication
- Strassen's Algorithm: Reduces multiplication count for large matrices
- Replication Hoisting: Moves replication operations to optimal positions
- Layout Hoisting: Optimizes layout assignments

These optimizations are applied after layout assignment to further
improve the efficiency of the generated FHE circuits.
"""

from .opt import Optimizer
from .roll_propagation import run_roll_propogation
from .roll_reordering import run_roll_reordering
from .bsgs_matmul import run_bsgs_matmul
from .strassens import run_strassens
from .replication_hoisting import run_replication_hoisting
from .layout_hoisting import run_layout_hoisting

__all__ = [
    'Optimizer', 'run_roll_propogation', 'run_roll_reordering',
    'run_bsgs_matmul', 'run_strassens', 'run_replication_hoisting',
    'run_layout_hoisting'
]
