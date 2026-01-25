"""
Test suite for compaction operations.

This module tests compaction operations that move dimensions from ct_dims to slot_dims
in the Rotom homomorphic encryption system.
"""

import numpy as np

from frontends.tensor import TensorTerm
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from lower.lower import Lower
from tests.conftest import assert_results_equal, run_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestCompaction:
    """Test compaction operations."""

    def _run_test_case(self, kernel, inputs, args, backend):
        """Helper method to run a test case."""
        # Run compiler
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)

        # Generate expected result from the tensor term
        tensor_term = kernel.cs[0].layout.term
        expected = tensor_term.eval(inputs)

        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert_results_equal(expected_cts, results, backend)

    def test_compact(self, backend):
        """Test simple compaction operation"""
        # Create args
        args = get_default_args()
        args.n = 16

        # Create inputs - 4x4 matrix
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )

        # Create tensor term
        a_term = TensorTerm.Tensor("a", [size, size], True)

        # Create prior layout: [1:4:1];[0:4:1][G:4]
        # This has dim 1 in ct_dims (extent 4, so 4 ciphertexts)
        prior_layout = Layout.from_string("[1:4:1];[0:4:1][G:4]", args.n, True)
        prior_layout.term = a_term

        # Create target layout: [0:4:1][1:4:1]
        # This has dim 1 moved to slot_dims
        target_layout = Layout.from_string("[0:4:1][1:4:1]", args.n, True)
        target_layout.term = a_term

        # Create input kernel (TENSOR)
        input_kernel = Kernel(KernelOp.TENSOR, [], prior_layout)

        # Create compaction kernel
        compact_kernel = Kernel(KernelOp.COMPACT, [input_kernel], target_layout)

        self._run_test_case(compact_kernel, inputs, args, backend)
