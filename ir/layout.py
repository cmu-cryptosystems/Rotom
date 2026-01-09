"""Layout class for packing tensor elements into HE vectors."""

import re

from ir.dim import Dim, DimType
from ir.roll import Roll
from util.util import prod


class Layout:
    """Layout class for packing tensor elements into HE vectors.

    The Layout class defines how tensor elements are packed into HE vectors.
    It handles dimensions, rolls, and offsets.

    Attributes:
        term: The tensor term this layout is associated with
        rolls: List of Roll permutations applied to the layout
        dims: List of Dim objects defining the dimensions and their order in the layout
        offset: Dictionary mapping dimensions to slot offsets
        n: Number of slots in the HE vector
        secret: Boolean indicating if this is a ciphertext (True) or plaintext (False)
        alignment: Dictionary storing dimension alignments
        ct_dims: List of dimensions mapped across ciphertexts
        slot_dims: List of dimensions mapped within slots

    """

    def __init__(self, term, rolls, dims, offset, n, secret=True):
        """Create a Layout

        Args:
            term: tensor term
            rolls: rolls applied to layout
            dims: dimensions (and order) of the layout
            n: number of slots
            secret: whether the layout is  a ciphertext or plaintext
        """
        self.term = term
        self.rolls = rolls
        self.offset = offset
        self.dims = dims
        self.n = n
        self.secret = secret
        self.alignment = {}
        if term and hasattr(term, "alignment") and term.alignment:
            self.alignment = term.alignment

        # check that there are no duplicate permutations
        assert len(rolls) == len(set(rolls))

        # offset is a dictionary of the form {dim: offset}
        # where offset is the number of slots to shift the dimension by
        assert isinstance(self.offset, dict)

        # # filter dimensions with extent <= 1
        # dims = [dim for dim in dims if dim.extent > 1]

        # split `dims` into ct_dims and slot_dims
        self.ct_dims = []
        self.slot_dims = []
        for dim in self.dims[::-1]:
            if not dim.extent:
                continue
            if n <= 1:
                self.ct_dims.insert(0, dim)
            elif dim.extent > n:
                assert dim.extent % n == 0
                slot_split_dim = Dim(dim.dim, n, dim_type=dim.dim_type)
                ct_split_dim = Dim(dim.dim, dim.extent // n, n, dim_type=dim.dim_type)
                self.slot_dims.insert(0, slot_split_dim)
                self.ct_dims.insert(0, ct_split_dim)
                n //= dim.extent
            elif dim.extent == n:
                self.slot_dims.insert(0, dim)
                n //= dim.extent
            else:
                assert n % dim.extent == 0
                n //= dim.extent
                self.slot_dims.insert(0, dim)
        if n > 1:
            self.slot_dims.insert(0, Dim.parse(f"G:{n}"))

        # assert no duplicate dimensions are added
        seen_dims = set()
        for dim in self.ct_dims + self.slot_dims:
            # assert dim not in seen_dims
            seen_dims.add(dim)

        # assert no duplicate strides are added
        seen_strides = {}
        for dim in self.ct_dims + self.slot_dims:
            if dim.dim not in seen_strides:
                seen_strides[dim.dim] = set()
            # assert dim.stride not in seen_strides[dim.dim]
            seen_strides[dim.dim].add(dim.stride)

        # assert rolls are all in dims
        all_dims = self.ct_dims + self.slot_dims
        for roll in self.rolls:
            assert roll.dim_to_roll in all_dims
            assert roll.dim_to_roll_by in all_dims

    def parse(self, layout_str):
        """Parse a layout string into a Layout object"""
        pattern = r"(roll\(\d+,\d+\)|\[\d+:\d+(:\d+)?\])"
        matches = re.findall(pattern, layout_str)

        # Extract only the full matches
        parsed = [match[0] if isinstance(match, tuple) else match for match in matches]

        rolls = []
        dims = []
        for item in parsed:
            if "roll" not in item:
                dims.append(Dim.parse(item))
        for item in parsed:
            if "roll" in item:
                pattern = r"roll\((\d+),(\d+)\)"
                match = re.match(pattern, item)
                roll = tuple(map(int, match.groups()))
                rolls.append(Roll(dims[roll[0]], dims[roll[1]]))
        return Layout(None, rolls, dims, 0, False)

    def layout_str(self):
        """String representation of the layout"""
        ct_dims = "".join([str(dim) for dim in self.ct_dims])
        slot_dims = "".join([str(dim) for dim in self.slot_dims])

        dims = self.ct_dims + self.slot_dims
        rolls = []
        for perm in self.rolls:
            rolls.append(
                f"roll({dims.index(perm.dim_to_roll)},{dims.index(perm.dim_to_roll_by)})"
            )
        rolls = " ".join(rolls)
        offset = " ".join([f"{k}:{v}" for k, v in self.offset.items()])
        layout = f"{ct_dims};{slot_dims}" if ct_dims else f"{slot_dims}"
        if rolls:
            layout = f"{rolls} {layout}"
        if offset:
            layout = f"{layout} {offset}"
        return layout

    def __repr__(self):
        """String representation of the layout"""
        layout_str = self.layout_str()
        return f"{layout_str}: {self.term}"

    def __lt__(self, other):
        """Compare two layouts"""
        if len(self.dims) != len(other.dims):
            return len(self.dims) < len(other.dims)
        if len(self.rolls) != len(other.rolls):
            return len(self.rolls) < len(other.rolls)
        return hash(self) < hash(self)

    def __hash__(self):
        """Hash the layout"""
        return hash(str(self))

    def __eq__(self, other):
        """Check if two layouts are equal"""
        return hash(self) == hash(other)

    def __len__(self):
        """Return the extent of all traversal dimensions in the layout"""
        return int(prod([dim.extent for dim in self.get_dims()]))

    def num_ct(self):
        """Return the number of ciphertexts used to represent the layout"""
        return int(prod([dim.extent for dim in self.ct_dims]))

    def num_ct_unique(self):
        """Return the number of unique ciphertexts used to represent the layout"""
        return int(prod([dim.extent for dim in self.ct_dims if dim.dim is not None]))

    def num_dims(self):
        """Return the number of ciphertext and slot dimensions"""
        return len(self.ct_dims) + len(self.slot_dims)

    def get_dims(self):
        """Return the list of ciphertext and slot dimensions"""
        return self.ct_dims + self.slot_dims

    @staticmethod
    def from_string(layout_str, n, secret=False):
        """
        Create a Layout object from a layout string.

        Args:
            layout_str: String representation of the layout (e.g., "roll(0,1) [1:4:1][0:4:1]")
            n: Number of slots in the HE vector
            secret: Whether this is ciphertext (True) or plaintext (False)

        Returns:
            Layout: The created layout object

        Examples:
            >>> layout = Layout.from_string("[0:4:1][1:4:1]", 16)
            >>> layout = Layout.from_string("roll(0,1) [1:4:1][0:4:1]", 16)
            >>> layout = Layout.from_string("[R:4:1];[0:4:1][1:4:1]", 16)
        """
        # Parse rolls from the layout string
        rolls = []
        dims_str = layout_str

        # Extract roll operations
        roll_pattern = r"roll\(([^,)]+),([^)]+)\)"
        roll_matches = re.findall(roll_pattern, layout_str)

        # Remove roll operations from the dimension string
        dims_str = re.sub(r"roll\([^)]+\)\s*", "", layout_str).strip()

        # Parse dimensions
        dims = []

        # Handle ciphertext dimensions (prefixed with [R: or ;)
        ct_dims_str = ""
        if ";" in dims_str:
            parts = dims_str.split(";")
            ct_dims_str = parts[0].strip()
            dims_str = parts[1].strip()

        # Parse ciphertext dimensions
        if ct_dims_str:
            ct_dim_matches = re.findall(r"\[([^\]]+)\]", ct_dims_str)
            for match in ct_dim_matches:
                if match.startswith("R:"):
                    # Handle replication dimension
                    extent = int(match.split(":")[1])
                    dims.append(Dim(None, extent, 1, DimType.FILL))
                elif match.startswith("G:"):
                    # Handle empty dimension
                    extent = int(match.split(":")[1])
                    dims.append(Dim(None, extent, 1, DimType.EMPTY))
                else:
                    # Handle regular dimension (e.g., [1:4:1])
                    dims.append(Dim.parse(f"[{match}]"))

        # Parse slot dimensions
        slot_dim_matches = re.findall(r"\[([^\]]+)\]", dims_str)
        for match in slot_dim_matches:
            dims.append(Dim.parse(f"[{match}]"))

        # Note: Roll parsing is deferred because we need the dimensions first
        # We'll store the roll indices and create the rolls after parsing dimensions
        roll_indices = []
        for from_str, to_str in roll_matches:
            from_idx = int(from_str)
            to_idx = int(to_str)
            roll_indices.append((from_idx, to_idx))

        # Create roll objects using the parsed dimensions
        for from_idx, to_idx in roll_indices:
            if from_idx < len(dims) and to_idx < len(dims):
                rolls.append(Roll(dims[from_idx], dims[to_idx]))

        # Create and return the layout
        return Layout(None, rolls, dims, {}, n, secret)
