"""
Homomorphic Encryption intermediate representation in Rotom.

The HE IR is used to represent homomorphic encryption operations and terms
in the Rotom intermediate representation. Rotom's layout IR is lowered to
the HE IR for execution on various HE backends.

Key Components:

- HEOp: Enumeration of supported homomorphic encryption operations
- HETerm: Representation of HE terms with operations and metadata
- Operation counting and analysis for cost estimation
- Post-order traversal for term evaluation
"""

from enum import Enum

from .layout_utils import dimension_merging


class HEOp(Enum):
    """
    Enumeration of homomorphic encryption operations.

    Defines all the HE operations supported by Rotom,
    including basic arithmetic, rotation, packing, and masking operations.
    These operations form the building blocks for more complex tensor
    computations in the homomorphic encryption domain.

    Operations:
        CS: Children term
        CS_PACK: Children term with defined packing
        PACK: Pack tensor data into HE vector
        INDICES: Index operations
        ADD: Homomorphic addition
        SUB: Homomorphic subtraction
        MUL: Homomorphic multiplication
        ROT: Rotation operation
        MASK: Masking operation
        POLY: Polynomial evaluation (WIP)
        RESCALE: Rescaling operation (WIP)
        ZERO_MASK: Zero masking
    """

    CS = "CS"
    CS_PACK = "CS_PACK"
    PACK = "PACK"
    TOEPLITZ_PACK = "TOEPLITZ_PACK"
    INDICES = "INDICES"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    ROT = "ROT"
    MASK = "MASK"
    POLY = "POLY"
    RESCALE = "RESCALE"
    ZERO_MASK = "ZERO_MASK"


class HETerm:
    """
    Represents a homomorphic encryption term in the IR.

    A HETerm encapsulates a single HE operation with its operands,
    metadata, and secret status. It forms the basic building block
    of the HE computation graph and supports HE operations like addition,
    multiplication, and rotation.

    Attributes:
        op: The HE operation type (from HEOp enum)
        cs: List of child terms (operands)
        secret: Boolean indicating if this term contains secret data
        metadata: Additional metadata string for the operation
        hash: Computed hash for term identity and comparison
    """

    def __init__(self, op, cs, secret, metadata=""):
        """
        Create a homomorphic encryption term.

        Args:
            op: The HE operation type
            cs: List of child terms (operands)
            secret: Whether this term contains secret data
            metadata: Additional metadata string

        Raises:
            ValueError: If cs is a tuple instead of a list
            AssertionError: If secret is not a boolean
        """
        self.op = op
        self.cs = cs
        self.metadata = metadata

        assert isinstance(secret, bool)
        self.secret = secret

        if isinstance(self.cs, tuple):
            raise ValueError("Child terms must be a list, not a tuple")

        cs_hashes = [c.hash if isinstance(c, HETerm) else c for c in self.cs]
        if self.op == HEOp.PACK or self.op == HEOp.CS_PACK or self.op == HEOp.CS or self.op == HEOp.TOEPLITZ_PACK:
            self.hash = hash(f"{self.op}:{cs_hashes}:{self.metadata}")
        else:
            self.hash = hash(f"{self.op}:{cs_hashes}")

    def ops(self):
        """Count the number of HE operations in this term.

        Returns:
            dict: Dictionary mapping operation types to their counts
        """
        count = {}
        for term in self.post_order():
            match term.op:
                case HEOp.ADD | HEOp.MUL | HEOp.ROT:
                    if term.secret:
                        if term.op not in count:
                            count[term.op] = 0
                        count[term.op] += 1
        return count

    def __hash__(self):
        """Compute hash of the HE term.

        Returns:
            int: Hash value for the term
        """
        return self.hash

    def __eq__(self, other):
        """Check equality of two HE terms.

        Args:
            other: Another HE term to compare with

        Returns:
            bool: True if terms are equal, False otherwise
        """
        return self.hash == other.hash

    def __add__(self, other):
        """Homomorphic addition operator.

        Args:
            other: The HE term to add

        Returns:
            HETerm: A new HE term representing the addition
        """
        return HETerm(HEOp.ADD, [self, other], self.secret or other.secret)

    def __sub__(self, other):
        """Homomorphic subtraction operator.

        Args:
            other: The HE term to subtract

        Returns:
            HETerm: A new HE term representing the subtraction
        """
        return HETerm(HEOp.SUB, [self, other], self.secret or other.secret)

    def __mul__(self, other):
        """Homomorphic multiplication operator.

        Args:
            other: The HE term to multiply

        Returns:
            HETerm: A new HE term representing the multiplication
        """
        return HETerm(HEOp.MUL, [self, other], self.secret or other.secret)

    def __lshift__(self, other):
        """Homomorphic rotation operator.

        Args:
            other: The rotation amount

        Returns:
            HETerm: A new HE term representing the rotation
        """
        return HETerm(HEOp.ROT, [self, other], self.secret)

    def pack(layout, metadata):
        """Create a pack operation term.

        Args:
            layout: The layout to pack
            metadata: Additional metadata for the operation

        Returns:
            HETerm: A new HE term representing the pack operation
        """
        return HETerm(HEOp.PACK, [layout], layout.secret, metadata)

    def mask(mask):
        """Create a mask operation term.

        Args:
            mask: The mask to apply

        Returns:
            HETerm: A new HE term representing the mask operation
        """
        return HETerm(HEOp.MASK, mask, False)

    def format_metadata(self, instr_str):
        """Format instruction string with metadata.

        Args:
            instr_str: The instruction string to format

        Returns:
            str: Formatted instruction string with metadata comment
        """
        if self.metadata:
            return f"{instr_str} # {self.metadata.split()[0]}"
        return instr_str

    def instrs(self, env={}, kernel_env={}):
        """Generate instruction strings for HE operations.

        Args:
            env: Environment mapping terms to indices
            kernel_env: Kernel environment for layout operations

        Returns:
            tuple: (list of instruction strings, updated environment)
        """
        idx = len(env)
        instruction_strs = []
        for term in self.post_order():
            if term in env:
                continue
            match term.op:
                case HEOp.CS:
                    he_term = term.cs[0]
                    match he_term.op:
                        case HEOp.CS_PACK:
                            layout_term = dimension_merging(he_term.cs[1])
                            assert layout_term in kernel_env
                            instruction_strs.append(
                                f"{idx} {term.secret}: {kernel_env[layout_term][he_term.cs[0]]}"
                            )
                        case HEOp.PACK:
                            instruction_strs.append(
                                he_term.format_metadata(
                                    f"{idx} {he_term.secret}: pack ({he_term.cs[0].layout_str()})"
                                )
                            )
                        case _:
                            pass
                            # raise NotImplementedError(he_term.op)
                case HEOp.PACK:
                    instruction_strs.append(
                        term.format_metadata(
                            f"{idx} {term.secret}: pack ({term.cs[0].layout_str()})"
                        )
                    )
                case HEOp.TOEPLITZ_PACK:
                    instruction_strs.append(
                        term.format_metadata(
                            f"{idx} {term.secret}: toeplitz pack ({term.cs[0].layout_str()})"
                        )
                    )
                case HEOp.CS_PACK:
                    layout_term = dimension_merging(term.cs[1])
                    assert layout_term in kernel_env
                    instruction_strs.append(
                        f"{idx} {term.secret}: {kernel_env[layout_term][term.cs[0]]}"
                    )
                case HEOp.MASK:
                    instruction_strs.append(
                        term.format_metadata(f"{idx} {term.secret}: mask {term.cs[0]}")
                    )
                case HEOp.ZERO_MASK:
                    instruction_strs.append(
                        term.format_metadata(f"{idx} {term.secret}: zero mask")
                    )
                case HEOp.ADD:
                    a = env[term.cs[0]]
                    b = env[term.cs[1]]
                    instruction_strs.append(f"{idx} {term.secret}: (+ {a} {b})")
                case HEOp.SUB:
                    a = env[term.cs[0]]
                    b = env[term.cs[1]]
                    instruction_strs.append(f"{idx} {term.secret}: (- {a} {b})")
                case HEOp.MUL:
                    a = env[term.cs[0]]
                    b = env[term.cs[1]]
                    instruction_strs.append(f"{idx} {term.secret}: (* {a} {b})")
                case HEOp.ROT:
                    a = env[term.cs[0]]
                    b = str(term.cs[1])
                    instruction_strs.append(f"{idx} {term.secret}: (<< {a} {b})")
                case HEOp.POLY:
                    a = env[term.cs[0]]
                    instruction_strs.append(f"{idx} {term.secret}: (poly {a})")
                case HEOp.RESCALE:
                    a = env[term.cs[0]]
                    b = term.cs[1]
                    instruction_strs.append(
                        f"{idx} {term.secret}: (rescale {a} / 2^{b})"
                    )
                case _:
                    raise NotImplementedError(term.op)
            env[term] = idx
            idx += 1
        return instruction_strs, env

    def __repr__(self):
        """String representation of the HE term.

        Returns:
            str: Human-readable representation of the term
        """
        match self.op:
            case HEOp.MASK:
                return f"{self.op} {self.cs[0]} {self.metadata}"
            case HEOp.ROT:
                return f"{self.op} {self.cs[1]} {self.metadata}"
            case HEOp.RESCALE:
                return f"{self.op} {self.cs[0]} / 2^{self.cs[1]} {self.metadata}"
            case _:
                return f"{self.op} {self.metadata}"

    def helper_post_order(self, seen):
        """Helper method for post-order traversal.

        Args:
            seen: Set of already visited nodes

        Returns:
            list: List of nodes in post-order
        """
        if self in seen:
            return []
        seen.add(self)
        match self.op:
            case HEOp.PACK | HEOp.TOEPLITZ_PACK | HEOp.INDICES | HEOp.CS | HEOp.CS_PACK:
                return [self]
            case HEOp.ADD | HEOp.SUB | HEOp.MUL:
                a = self.cs[0].helper_post_order(seen)
                b = self.cs[1].helper_post_order(seen)
                return a + b + [self]
            case HEOp.ROT | HEOp.POLY | HEOp.RESCALE:
                a = self.cs[0].helper_post_order(seen)
                return a + [self]
            case HEOp.MASK | HEOp.ZERO_MASK:
                return [self]
            case _:
                raise NotImplementedError(self.op)

    def post_order(self):
        """Perform post-order traversal of the HE computation DAG.

        Returns:
            list: List of HE terms in post-order (children before parents)
        """
        return self.helper_post_order(set())
