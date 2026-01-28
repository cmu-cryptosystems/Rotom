from frontends.tensor import TensorTerm

from ..dim import Dim
from ..kernel import Kernel, KernelOp
from ..kernel_cost import KernelCost
from ..layout import Layout
from ..roll import Roll


# Dimensions
def test_layout_dims_1():
    n = 4
    A_layout = Layout(None, [], [Dim.parse("[0:4:1]")], n)
    assert A_layout.layout_str() == "[0:4:1]"


def test_layout_dims_2():
    n = 4
    A_layout = Layout(
        None, [], [Dim.parse("[0:4:1]"), Dim.parse("[4]"), Dim.parse("[1:4:1]")], n
    )
    assert A_layout.layout_str() == "[0:4:1][R:4:1];[1:4:1]"


def test_layout_dims_3():
    n = 16
    A_layout = Layout(
        None, [], [Dim.parse("[0:4:1]"), Dim.parse("[4]"), Dim.parse("[1:4:1]")], n
    )
    assert A_layout.layout_str() == "[0:4:1];[R:4:1][1:4:1]"


def test_layout_dims_4():
    n = 16
    A_layout = Layout(
        None, [], [Dim.parse("[0:8:1]"), Dim.parse("[8]"), Dim.parse("[1:8:1]")], n
    )
    assert A_layout.layout_str() == "[0:8:1][R:4:2];[R:2:1][1:8:1]"


# Replication
def test_cost_replication_1():
    n = 4
    A_layout = Layout(TensorTerm.Tensor("a", [], True), [], [Dim.parse("[0:4:1]")], n)
    B_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[4]"), Dim.parse("[0:4:1]")],
        n,
    )

    A_term = Kernel(KernelOp.TENSOR, [], A_layout)
    layout_ir = Kernel(KernelOp.REPLICATE, [A_term], B_layout)

    # Replicating by copying cts is free
    operations = KernelCost(layout_ir, "lan").total_operations()
    assert operations["mul"] == 0
    assert operations["add"] == 0
    assert operations["rot"] == 0


def test_cost_replication_2():
    n = 4
    A_layout = Layout(TensorTerm.Tensor("a", [], True), [], [Dim.parse("[0:4:1]")], n)
    B_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[4]")],
        n,
    )

    A_term = Kernel(KernelOp.TENSOR, [], A_layout)
    layout_ir = Kernel(KernelOp.REPLICATE, [A_term], B_layout)

    operations = KernelCost(layout_ir, "lan").total_operations()
    assert operations["mul"] == 4
    assert operations["add"] == 8
    assert operations["rot"] == 8


def test_cost_replication_3():
    n = 16
    A_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]")],
        n,
    )
    B_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]"), Dim.parse("[16]")],
        n,
    )

    A_term = Kernel(KernelOp.TENSOR, [], A_layout)
    layout_ir = Kernel(KernelOp.REPLICATE, [A_term], B_layout)

    operations = KernelCost(layout_ir, "lan").total_operations()
    assert operations["mul"] == 16
    assert operations["add"] == 64
    assert operations["rot"] == 64


def test_cost_replication_4():
    n = 16
    A_layout = Layout(TensorTerm.Tensor("a", [], True), [], [Dim.parse("[0:4:1]")], n)
    B_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[4]"), Dim.parse("[0:4:1]")],
        n,
    )

    A_term = Kernel(KernelOp.TENSOR, [], A_layout)
    layout_ir = Kernel(KernelOp.REPLICATE, [A_term], B_layout)

    operations = KernelCost(layout_ir, "lan").total_operations()
    assert operations["mul"] == 0
    assert operations["add"] == 2
    assert operations["rot"] == 2


# Tensor operations


def test_cost_matvecmul_1():
    n = 4
    M_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]")],
        n,
    )
    V_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[4]"), Dim.parse("[0:4:1]")],
        n,
    )
    R_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[4]")],
        n,
    )

    M_term = Kernel(KernelOp.TENSOR, [], M_layout)
    V_term = Kernel(KernelOp.TENSOR, [], V_layout)
    layout_ir = Kernel(KernelOp.MATMUL, [M_term, V_term], R_layout)

    # 1. mul == 4 because there are 4 cts
    # 2. there are no ct sum dimensions
    # 3. add == 8 because the length of the slot sum dimensions is 4, 4 * log2(4)
    # 4. rot == 8 because the length of the slot sum dimensions is 4, 4 * log2(4)

    operations = KernelCost(layout_ir, "lan").total_operations()
    assert operations["mul"] == 4
    assert operations["add"] == 8
    assert operations["rot"] == 8


def test_cost_matvecmul_2():
    n = 16
    M_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]")],
        n,
    )
    V_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[4]"), Dim.parse("[0:4:1]")],
        n,
    )
    R_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[4]")],
        n,
    )

    M_term = Kernel(KernelOp.TENSOR, [], M_layout)
    V_term = Kernel(KernelOp.TENSOR, [], V_layout)
    layout_ir = Kernel(KernelOp.MATMUL, [M_term, V_term], R_layout)

    # 1. mul == 1 because there is 1 ct
    # 2. there are no ct sum dimensions
    # 3. add == 2 because the length of the slot sum dimensions is 4, 1 * log2(4)
    # 4. rot == 2 because the length of the slot sum dimensions is 4, 1 * log2(4)

    operations = KernelCost(layout_ir, "lan").total_operations()
    assert operations["mul"] == 1
    assert operations["add"] == 2
    assert operations["rot"] == 2


def test_cost_matmul_1():
    n = 16
    A_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[1:4:1]"), Dim.parse("[0:4:1]"), Dim.parse("[4]")],
        n,
    )
    B_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[4]"), Dim.parse("[1:4:1]")],
        n,
    )
    C_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[4]"), Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]")],
        n,
    )

    A_term = Kernel(KernelOp.TENSOR, [], A_layout)
    B_term = Kernel(KernelOp.TENSOR, [], B_layout)
    layout_ir = Kernel(KernelOp.MATMUL, [A_term, B_term], C_layout)

    # 1. mul == 4, because there are 4 cts
    # 2. add == 4, because the length of the slot ct dimensions is 4
    # 3. add == 0, because the length of the slot sum dimensions is 0
    # 4. rot == 0, because the length of the slot sum dimensions is 0

    operations = KernelCost(layout_ir, "lan").total_operations()
    assert operations["mul"] == 4
    assert operations["add"] == 4
    assert operations["rot"] == 0


def test_cost_matmul_2():
    n = 4
    A_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[1:4:1]"), Dim.parse("[0:4:1]"), Dim.parse("[4]")],
        n,
    )
    B_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[4]"), Dim.parse("[1:4:1]")],
        n,
    )
    C_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[4]"), Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]")],
        n,
    )

    A_term = Kernel(KernelOp.TENSOR, [], A_layout)
    B_term = Kernel(KernelOp.TENSOR, [], B_layout)
    layout_ir = Kernel(KernelOp.MATMUL, [A_term, B_term], C_layout)

    # 1. mul == 16, because there are 16 cts
    # 2. add == 16, because the length of the slot ct dimensions is 4, and there are 16 cts
    # 3. add == 0, because the length of the slot sum dimensions is 0
    # 4. rot == 0, because the length of the slot sum dimensions is 0

    operations = KernelCost(layout_ir, "lan").total_operations()
    assert operations["mul"] == 16
    assert operations["add"] == 4
    assert operations["rot"] == 0


def test_cost_matmul_3():
    n = 16
    A_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[4]"), Dim.parse("[1:4:1]")],
        n,
    )
    B_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[4]"), Dim.parse("[1:4:1]"), Dim.parse("[0:4:1]")],
        n,
    )
    C_layout = Layout(
        TensorTerm.Tensor("a", [], True),
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]"), Dim.parse("[4]")],
        n,
    )

    A_term = Kernel(KernelOp.TENSOR, [], A_layout)
    B_term = Kernel(KernelOp.TENSOR, [], B_layout)
    layout_ir = Kernel(KernelOp.MATMUL, [A_term, B_term], C_layout)

    # 1. mul == 4, because there are 4 cts
    # 2. there are no ct sum dimensions
    # 3. add == 8, because the length of the slot sum dimensions is 4, 4 * log2(4)
    # 4. rot == 8, because the length of the slot sum dimensions is 4, 4 * log2(4)

    operations = KernelCost(layout_ir, "lan").total_operations()
    assert operations["mul"] == 4
    assert operations["add"] == 8
    assert operations["rot"] == 8


# Tests for Layout.from_string method
def test_layout_from_string_simple():
    """Test creating a layout from a simple string with just slot dimensions."""
    layout = Layout.from_string("[0:4:1]", 16)
    assert layout.layout_str() == "[G:4][0:4:1]"
    assert layout.n == 16
    assert layout.secret == False
    assert len(layout.rolls) == 0
    assert len(layout.dims) == 1


def test_layout_from_string_multiple_dimensions():
    """Test creating a layout from a string with multiple slot dimensions."""
    layout = Layout.from_string("[0:4:1][1:4:1]", 16)
    assert layout.layout_str() == "[0:4:1][1:4:1]"
    assert layout.n == 16
    assert layout.secret == False
    assert len(layout.rolls) == 0
    assert len(layout.dims) == 2


def test_layout_from_string_with_roll():
    """Test creating a layout from a string with roll operations."""
    layout = Layout.from_string("roll(0,1) [1:4:1][0:4:1]", 16)
    assert layout.layout_str() == "roll(0,1) [1:4:1][0:4:1]"
    assert layout.n == 16
    assert layout.secret == False
    assert len(layout.rolls) == 1
    assert len(layout.dims) == 2


def test_layout_from_string_with_ciphertext_dimensions():
    """Test creating a layout from a string with ciphertext dimensions."""
    layout = Layout.from_string("[R:4:1];[0:4:1][1:4:1]", 16)
    assert layout.layout_str() == "[R:4:1];[0:4:1][1:4:1]"
    assert layout.n == 16
    assert layout.secret == False
    assert len(layout.rolls) == 0
    assert len(layout.dims) == 3


def test_layout_from_string_with_roll_and_ciphertext():
    """Test creating a layout from a string with both roll operations and ciphertext dimensions."""
    layout = Layout.from_string("roll(0,1) [R:4:1];[1:4:1][0:4:1]", 16)
    assert layout.layout_str() == "roll(0,1) [R:4:1];[1:4:1][0:4:1]"
    assert layout.n == 16
    assert layout.secret == False
    assert len(layout.rolls) == 1
    assert len(layout.dims) == 3


def test_layout_from_string_secret_true():
    """Test creating a secret layout from a string."""
    layout = Layout.from_string("[0:4:1]", 16, secret=True)
    assert layout.layout_str() == "[G:4][0:4:1]"
    assert layout.n == 16
    assert layout.secret == True


def test_layout_from_string_complex_rolls():
    """Test creating a layout with multiple roll operations."""
    layout = Layout.from_string("roll(0,1) roll(2,3) [0:4:1][1:4:1][2:4:1][3:4:1]", 16)
    assert layout.layout_str() == "roll(0,1) roll(2,3) [0:4:1][1:4:1];[2:4:1][3:4:1]"
    assert layout.n == 16
    assert layout.secret == False
    assert len(layout.rolls) == 2
    assert len(layout.dims) == 4


def test_layout_from_string_edge_cases():
    """Test edge cases for layout string parsing."""
    # Test with extra whitespace
    layout = Layout.from_string("  [0:4:1]  ", 16)
    assert layout.layout_str() == "[G:4][0:4:1]"

    # Test with empty roll operations (should be ignored)
    layout = Layout.from_string("roll(,) [0:4:1]", 16)
    assert layout.layout_str() == "[G:4][0:4:1]"
    assert len(layout.rolls) == 0


def test_layout_from_string_parsing_accuracy():
    """Test that the parsed layout matches manually created layouts."""
    # Test simple case
    layout_str = "[0:4:1][1:4:1]"
    parsed_layout = Layout.from_string(layout_str, 16)
    manual_layout = Layout(
        None, [], [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]")], 16, False
    )

    assert parsed_layout.layout_str() == manual_layout.layout_str()
    assert parsed_layout.n == manual_layout.n
    assert parsed_layout.secret == manual_layout.secret

    # Test with roll
    layout_str = "roll(0,1) [1:4:1][0:4:1]"
    parsed_layout = Layout.from_string(layout_str, 16)
    # The roll indices refer to positions in the dimension list
    dims = [Dim.parse("[1:4:1]"), Dim.parse("[0:4:1]")]
    manual_layout = Layout(None, [Roll(dims[0], dims[1])], dims, 16, False)

    assert parsed_layout.layout_str() == manual_layout.layout_str()
    assert len(parsed_layout.rolls) == len(manual_layout.rolls)
    assert len(parsed_layout.dims) == len(manual_layout.dims)


def test_layout_from_string_invalid_input():
    """Test behavior with invalid input strings."""
    # Test with malformed dimension strings (should raise an exception)
    try:
        Layout.from_string("[invalid]", 16)
        assert False, "Should have raised an exception for invalid dimension"
    except Exception:
        pass  # Expected behavior

    # Test with malformed roll strings (should handle gracefully)
    layout = Layout.from_string("roll(invalid) [0:4:1]", 16)
    assert layout.layout_str() == "[G:4][0:4:1]"
    assert len(layout.rolls) == 0  # Invalid roll should be ignored


def test_layout_from_string_roundtrip():
    """Test that layout string generation and parsing are consistent."""
    test_cases = [
        "[0:4:1]",
        "[0:4:1][1:4:1]",
        "roll(0,1) [1:4:1][0:4:1]",
        "[R:4:1];[0:4:1][1:4:1]",
        "roll(0,1) [R:4:1];[1:4:1][0:4:1]",
    ]

    for layout_str in test_cases:
        # Create layout from string
        layout = Layout.from_string(layout_str, 16)
        # Get the string representation
        generated_str = layout.layout_str()
        # Parse the generated string again
        roundtrip_layout = Layout.from_string(generated_str, 16)
        # Compare layouts
        assert roundtrip_layout.layout_str() == layout.layout_str()
        assert roundtrip_layout.n == layout.n
        assert roundtrip_layout.secret == layout.secret
