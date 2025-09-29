"""
Dimension representation.

Dimensions are used to track tensor dimensions, their extents, strides, and
types. They are fundamental to representing how tensor data is packed into HE vectors.

Key Concepts:

- Dimension extent: The range of values a dimension can take
- Dimension stride: The step size when indexing through a dimension
- Dimension type: Whether a dimension is filled with data or empty (zeroed)
"""

import math
from enum import Enum

from frontends.tensor import *


class DimType(Enum):
    """
    Enumeration defining dimension types for tensor layouts.

    Dimensions can be either filled with actual data or empty (zero-filled).
    This distinction is important for optimization and memory management
    in homomorphic encryption contexts.

    Attributes:
        FILL: Dimension contains actual tensor data
        EMPTY: Dimension is zero-filled (used for padding)
    """

    FILL = 0  # filled dimension
    EMPTY = 1  # zero-filled dimension


class Dim:
    """
    Represents a tensor dimension with extent, stride, and type.

    Attributes:
        dim: The tensor dimension index this Dim represents
        extent: The range of values this dimension can take (must be power of 2)
        stride: The step size when indexing through this dimension (must be power of 2)
        dim_type: Whether this dimension is FILL or EMPTY
    """

    def __init__(self, dim, extent, stride=1, dim_type=DimType.FILL):
        """
        Create a dimension.

        Args:
            dim: Tensor dimension index to map to
            extent: Range of indexing (must be a power of 2)
            stride: Step size taken when indexing from [0, extent) (must be power of 2)
            dim_type: Type of dimension (FILL for data, EMPTY for zero-filled)

        Raises:
            AssertionError: If extent or stride are not powers of 2, or stride <= 0
        """
        self.dim = dim
        self.extent = extent
        self.stride = stride
        self.dim_type = dim_type

        if self.dim_type == DimType.EMPTY:
            assert self.dim is None

        assert math.log2(self.extent).is_integer()
        assert math.log2(self.stride).is_integer()
        assert self.stride > 0

    def __hash__(self):
        """Hash the dimension"""
        return hash(str(self))

    def __eq__(self, other):
        """Check if two dimensions are equal"""
        return hash(self) == hash(other)

    def __lt__(self, other):
        """Compare two dimensions"""
        return self.stride < other.stride

    def parse(string_repr):
        """Parser function to transform a string-represented Dimension
        into a Dimension object
        """
        terms = string_repr.replace("[", "").replace("]", "").split(":")
        if len(terms) == 1:
            return Dim(None, int(terms[0]), 1, dim_type=DimType.FILL)
        elif len(terms) == 2 and terms[0] == "G":
            return Dim(None, int(terms[1]), 1, dim_type=DimType.EMPTY)
        elif len(terms) == 3:
            return Dim(
                int(terms[0]), int(terms[1]), int(terms[2]), dim_type=DimType.FILL
            )
        else:
            raise NotImplementedError(string_repr)

    def __repr__(self):
        """String representation of the dimension"""
        match self.dim_type:
            case DimType.FILL:
                if self.dim is None:
                    return f"[R:{self.extent}:{self.stride}]"
                else:
                    return f"[{self.dim}:{self.extent}:{self.stride}]"
            case DimType.EMPTY:
                return f"[G:{self.extent}]"

    def copy(self):
        """Copy the dimension"""
        return Dim(
            self.dim,
            self.extent,
            self.stride,
            self.dim_type,
        )
