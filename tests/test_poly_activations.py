"""
Tests for Poly-based activation functions (SiLU, ReLU, etc.).
"""

import numpy as np

from frontends.tensor import TensorTerm


class TestPolyActivations:
    """Test Poly evaluation for activation-style funcs."""

    def test_poly_silu(self):
        """Poly with silu approximates x * sigmoid(x)."""
        a = TensorTerm.Tensor("a", [2, 2], True)
        out = a.poly("silu")
        x = np.array([[0.0, 1.0], [-1.0, 2.0]])
        inputs = {"a": x}
        result = out.eval(inputs)
        expected = x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)

    def test_poly_relu_chebyshev(self):
        """Poly with relu approximates max(0, x) via Chebyshev polynomial."""
        # Use a power-of-two-friendly shape to avoid padding effects.
        a = TensorTerm.Tensor("a", [2, 4], True)
        out = a.poly("relu")
        # Test on a small range where the Chebyshev fit is accurate.
        x = np.array([[-3.0, -1.0, 0.0, 1.5], [2.0, -0.5, 3.0, -2.0]])
        inputs = {"a": x}
        result = out.eval(inputs)
        expected = np.maximum(x, 0.0)
        # Allow a modest tolerance since this is an approximation polynomial.
        # The underlying Chebyshev fit is tuned for a bounded domain and a
        # moderate polynomial degree, so we only expect agreement up to a few
        # percent rather than machine precision.
        np.testing.assert_allclose(result, expected, rtol=5e-2, atol=5e-2)
