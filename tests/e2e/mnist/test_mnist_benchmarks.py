"""E2E-style tests that exercise the MNIST benchmarks.

These tests ensure that the shared MNIST data utilities in
`benchmarks.e2e.mnist.mnist_data` work correctly with the benchmark
entrypoints, and that we can successfully build/evaluate the tensor IR.
"""

from benchmarks.e2e.mnist.mnist_data import load_mnist_test_set
from benchmarks.e2e.mnist.mnist_poly import mnist_poly
from benchmarks.e2e.mnist.mnist_poly_call import mnist_poly_call


def test_mnist_data_loads():
    """Basic sanity check that MNIST test data can be loaded."""
    images, labels = load_mnist_test_set()
    assert images.shape[0] == labels.shape[0]
    assert images.shape[0] > 0


def test_mnist_poly_builds_tensor_ir():
    """Ensure mnist_poly benchmark builds a tensor IR without errors."""
    images, _ = load_mnist_test_set()
    tensor_ir, inputs, label = mnist_poly(idx=0)

    # Basic shape sanity checks
    assert "input" in inputs
    assert tensor_ir is not None
    assert isinstance(label, int)


def test_mnist_poly_call_builds_tensor_ir():
    """Ensure mnist_poly_call benchmark builds a tensor IR without errors."""
    images, _ = load_mnist_test_set()
    tensor_ir, inputs, label = mnist_poly_call(idx=0)

    assert "input" in inputs
    assert tensor_ir is not None
    assert isinstance(label, int)
