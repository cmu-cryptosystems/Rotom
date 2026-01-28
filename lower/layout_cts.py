"""Layout-aware ciphertext representation for lowering."""

from copy import copy
from typing import Dict, List

from ir.dim import Dim, DimType
from ir.he import HETerm
from ir.layout import Layout


class LayoutCiphertexts:
    """
    A layout-aware representation of ciphertexts.

    This class pairs a Layout with its corresponding ciphertexts,
    making it easier to understand which dimensions need to be
    summed or manipulated during lowering.

    Attributes:
        layout: The Layout describing how data is organized
        cts: Dictionary mapping ciphertext indices to HETerm objects
    """

    def __init__(self, layout: Layout, cts: Dict[int, HETerm]):
        """Create a LayoutCiphertexts object.

        Args:
            layout: The Layout describing how data is organized
            cts: Dictionary mapping ciphertext indices to HETerm objects
        """
        self.layout = layout
        self.cts = cts

    def __getitem__(self, key: int) -> HETerm:
        """Access ciphertext by index."""
        return self.cts[key]

    def keys(self):
        """Get ciphertext indices."""
        return self.cts.keys()

    def values(self):
        """Get ciphertext values."""
        return self.cts.values()

    def items(self):
        """Get (index, ciphertext) pairs."""
        return self.cts.items()

    def copy(self):
        """Create a shallow copy."""
        return LayoutCiphertexts(layout=copy(self.layout), cts=dict(self.cts))

    def update_layout(self, new_layout: Layout):
        """Update the layout, keeping the same ciphertexts."""
        return LayoutCiphertexts(layout=new_layout, cts=self.cts)

    def update_cts(self, new_cts: Dict[int, HETerm]):
        """Update the ciphertexts, keeping the same layout."""
        return LayoutCiphertexts(layout=self.layout, cts=new_cts)

    def __len__(self) -> int:
        """Number of ciphertexts."""
        return len(self.cts)

    def __repr__(self) -> str:
        return f"LayoutCiphertexts(layout={self.layout.layout_str()}, num_cts={len(self.cts)})"

    def __eq__(self, other):
        """Check equality."""
        if not isinstance(other, LayoutCiphertexts):
            return False
        return self.layout == other.layout and self.cts == other.cts


def create_layout_without_dims(layout: Layout, dims_to_remove: List[Dim]) -> Layout:
    """
    Create a new Layout without the specified dimensions.

    This function removes dimensions from both ct_dims and slot_dims,
    and filters out any rolls that involve the removed dimensions.

    Args:
        layout: The original layout
        dims_to_remove: List of Dim objects to remove

    Returns:
        A new Layout object without the removed dimensions
    """
    dims_to_remove_set = set(dims_to_remove)

    # Filter out removed dimensions from ct_dims and slot_dims
    new_ct_dims = [dim for dim in layout.ct_dims if dim not in dims_to_remove_set]
    new_slot_dims = [
        (
            Dim(None, dim.extent, dim.stride, DimType.EMPTY)
            if dim in dims_to_remove_set
            else dim
        )
        for dim in layout.slot_dims
    ]

    # Combine into full dims list (ct_dims come before slot_dims in the full dims)
    new_dims = new_ct_dims + new_slot_dims

    # Filter out rolls that involve removed dimensions
    new_rolls = []
    for roll in layout.rolls:
        if (
            roll.dim_to_roll not in dims_to_remove_set
            and roll.dim_to_roll_by not in dims_to_remove_set
        ):
            # Check if the roll dimensions still exist in new_dims
            if roll.dim_to_roll in new_dims and roll.dim_to_roll_by in new_dims:
                new_rolls.append(roll)

    # Create new layout with filtered dimensions and rolls
    return Layout(layout.term, new_rolls, new_dims, layout.n, layout.secret)
