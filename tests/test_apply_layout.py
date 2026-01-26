"""
Test suite for apply_layout function.

This module tests the apply_layout function which applies a layout to a plaintext tensor
to pack it into ciphertext vectors.
"""

import numpy as np

from ir.dim import Dim, DimType
from ir.layout import Layout
from util.layout_util import apply_layout


class TestApplyLayout:
    """Test apply_layout function with various layouts."""

    def test_apply_layout_4x4x4_tensor(self):
        """Test apply_layout with a 4x4x4 tensor and complex layout."""
        # Create the input tensor: 4x4x4 array with values 0-15
        # The tensor has shape (4, 4, 4) where each 4x4 slice contains values 0-15
        tensor = np.array(
            [
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            ]
        )

        # Create layout manually: [1:2:8][1:2:4][0:2:2][0:2:1];[R:2:2][R:2:1][2:2:2][2:2:1]
        # CT dims: [1:2:8][1:2:4][0:2:2][0:2:1]
        # Slot dims: [R:2:2][R:2:1][2:2:2][2:2:1]
        dims = [
            # CT dimensions
            Dim(1, 2, 2, DimType.FILL),  # [1:2:8]
            Dim(1, 2, 1, DimType.FILL),  # [1:2:4]
            Dim(0, 2, 2, DimType.FILL),  # [0:2:2]
            Dim(0, 2, 1, DimType.FILL),  # [0:2:1]
            # Slot dimensions
            Dim(None, 2, 2, DimType.FILL),  # [R:2:2]
            Dim(None, 2, 1, DimType.FILL),  # [R:2:1]
            Dim(2, 2, 2, DimType.FILL),  # [2:2:2]
            Dim(2, 2, 1, DimType.FILL),  # [2:2:1]
        ]
        layout = Layout(None, [], dims, {}, n=16, secret=False)

        # Apply layout
        result = apply_layout(tensor, layout)

        # Expected result from terminal output
        expected = [
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        # Convert result to list of lists for comparison
        result_list = [list(ct) for ct in result]

        # Assert the result matches expected
        assert len(result_list) == len(expected), f"Expected {len(expected)} ciphertexts, got {len(result_list)}"
        for i, (result_ct, expected_ct) in enumerate(zip(result_list, expected)):
            assert len(result_ct) == len(expected_ct), f"Ciphertext {i}: expected length {len(expected_ct)}, got {len(result_ct)}"
            assert result_ct == expected_ct, f"Ciphertext {i} mismatch:\n  Expected: {expected_ct}\n  Got:      {result_ct}"
