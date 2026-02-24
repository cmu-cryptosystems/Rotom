"""
E2E test: MLP-on-MNIST style inference (two FC layers, linear only).

Based on https://github.com/google/heir/issues/1232: MLP inference on MNIST
using HEIR CKKS. The HEIR write-up uses FC1 (1024x1024), approx-RELU, FC2 (1024x1024).
Rotom does not implement ReLU in HE, so we test the linear part only:
  output = (input @ fc1) @ fc2.

Tests use small dimensions for fast execution; the full 1024x1024 variant
can be run via: python main.py --benchmark mlp_mnist --backend ckks --rolls ...
"""

import numpy as np

from assignment.assignment import LayoutAssignment
from benchmarks.rotom_benchmarks.mlp_mnist import mlp_mnist
from lower.lower import Lower
from tests.conftest import assert_results_equal, run_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestMlpMnist:
    """Test MLP-MNIST style two-layer linear inference (input @ fc1 @ fc2)."""

    def _run_test_case(self, tensor_ir, inputs, args, backend):
        """Run layout assignment, lower, execute backend, and check results."""
        expected = tensor_ir.eval(inputs)
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)
        expected_cts = apply_layout(expected, kernel.layout)
        # MLP uses float inputs; toy returns floats (0.0, -0.0) so use allclose for both
        if backend == "toy":
            for exp, res in zip(expected_cts, results):
                assert np.allclose(
                    np.asarray(exp), np.asarray(res), rtol=1e-9, atol=1e-9
                ), f"Toy results not close. Expected: {exp}, Got: {res}"
        else:
            assert_results_equal(expected_cts, results, backend)

    def test_mlp_mnist_4x4(self, backend):
        """MLP with 1x4 input and 4x4 weight matrices (tiny, for sanity check)."""
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "mlp_mnist_4"
        tensor_ir, inputs = mlp_mnist(hidden_size=4)
        self._run_test_case(tensor_ir, inputs, args, backend)

    def test_mlp_mnist_16x16(self, backend):
        """MLP with 1x16 input and 16x16 weight matrices (small scale)."""
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "mlp_mnist_16"
        tensor_ir, inputs = mlp_mnist(hidden_size=16)
        self._run_test_case(tensor_ir, inputs, args, backend)

    def test_mlp_mnist_64x64(self, backend):
        """MLP with 1x64 input and 64x64 weight matrices (medium scale)."""
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "mlp_mnist_64"
        tensor_ir, inputs = mlp_mnist(hidden_size=64)
        self._run_test_case(tensor_ir, inputs, args, backend)
