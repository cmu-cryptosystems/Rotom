"""
Roll operations for tensor dimension permutations.

This module provides roll operations that represent modular addition
between tensor dimensions. Rolls are fundamental to optimizing tensor
operations in homomorphic encryption by enabling efficient data
rearrangement and rotation operations.

Key Concepts:

- Roll operation: Modular addition between two dimensions
- Dimension rolling: Moving data between dimension positions
- Roll optimization: Minimizing expensive rotation operations
- Roll propagation: Moving rolls to optimal positions
"""

from .dim import Dim


class Roll:
    """
    Represents a roll operation between two tensor dimensions.

    A roll operation performs modular addition between the indices of
    two dimensions, effectively rotating data between dimension positions.
    This is crucial for optimizing tensor operations in homomorphic
    encryption where rotation operations are expensive.

    Attributes:
        dim_to_roll: The dimension being rolled (result dimension)
        dim_to_roll_by: The dimension used for rolling (additive dimension)
    """

    def __init__(self, dim_to_roll, dim_to_roll_by):
        """Initialize a roll operation"""
        assert dim_to_roll != dim_to_roll_by
        assert isinstance(dim_to_roll, Dim)
        assert isinstance(dim_to_roll_by, Dim)
        assert dim_to_roll.extent == dim_to_roll_by.extent

        self.dim_to_roll = dim_to_roll
        self.dim_to_roll_by = dim_to_roll_by

    def __repr__(self):
        """String representation of the roll operation"""
        return f"roll({self.dim_to_roll},{self.dim_to_roll_by})"

    def __eq__(self, other):
        """Check if two roll operations are equal"""
        return hash(self) == hash(other)

    def __hash__(self):
        """Hash the roll operation"""
        return hash(str(self))

    def copy(self):
        """Copy the roll operation"""
        return Roll(self.dim_to_roll, self.dim_to_roll_by)

    def roll_index(self, dims):
        """Get the index of the roll operation"""
        assert self.dim_to_roll in dims
        assert self.dim_to_roll_by in dims
        return (dims.index(self.dim_to_roll), dims.index(self.dim_to_roll_by))

    def roll_update(self, dims):
        """Update the roll operation"""
        assert self.dim_to_roll in dims
        assert self.dim_to_roll_by in dims
        return (self.dim_to_roll, dims.index(self.dim_to_roll_by))
