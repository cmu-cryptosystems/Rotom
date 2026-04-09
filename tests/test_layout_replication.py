import numpy as np

from util.layout_util import apply_layout, parse_layout


def test_apply_layout_with_replicated_slot_dim():
    # Simple 1D tensor
    pt = np.array([1, 2])

    # Layout with a replicated dimension that ends up in slot_dims.
    # For n=4 this produces slot_dims = [R:2:1][0:2:1] and no ct_dims.
    layout = parse_layout("[R:2:1][0:2:1]", secret=False)

    # Sanity-check that we really have a replicated slot dimension
    assert layout.num_ct() == 0 or layout.num_ct() == 1
    assert any(dim.dim is None for dim in layout.slot_dims)

    cts = apply_layout(pt, layout)

    # One ciphertext with 4 slots: base pattern [1, 2] replicated over slots
    assert len(cts) == max(1, layout.num_ct())
    np.testing.assert_array_equal(cts[0], [1, 2, 1, 2])


def test_apply_layout_with_replication_every_16_slots():
    # Input tensor of shape [1, 3, 3, 3] with distinct values 1..27.
    pt = np.arange(1, 28, dtype=int).reshape(1, 3, 3, 3)

    # Layout:
    #   ct dims:  [2:3:1][3:3:1]   -> 9 ciphertexts, one per (h, w)
    #   slot dims:[1:4:1][R:16:1]  -> 4 base slots (dim 1), each replicated 16 times (R)
    # So each CT has 4 blocks of 16: value at dim1=0 repeated 16×, then dim1=1 ×16, etc.
    layout = parse_layout("[2:3:1][3:3:1];[1:4:1][R:16:1]", secret=False)

    assert layout.num_ct() == 9
    assert any(dim.dim is None for dim in layout.slot_dims)

    cts = apply_layout(pt, layout)
    assert len(cts) == layout.num_ct()

    # First CT (h=0, w=0): base slots are pt[0, 0, 0, 0]=1, pt[0, 1, 0, 0]=10, pt[0, 2, 0, 0]=19, dim1=3 OOB -> 0.
    first_ct = cts[0]
    assert first_ct[0] == 1

    # R:16 replicates each base slot 16 times: block 0 = 1×16, block 1 = 10×16, block 2 = 19×16, block 3 = 0×16.
    block_len = 16
    num_blocks = 64 // block_len
    expected_block_values = [1, 10, 19, 0]
    for k in range(num_blocks):
        for j in range(block_len):
            assert first_ct[k * block_len + j] == expected_block_values[k]
