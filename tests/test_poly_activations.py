"""
Tests for Poly-based activation functions (SiLU, ReLU, etc.).
"""

import numpy as np

from frontends.tensor import TensorTerm


class TestPolyEval:
    """Test Poly evaluation for generic funcs (identity, explicit coeffs)."""

    def test_poly_identity(self):
        """Poly with identity returns input unchanged (up to power-of-2 padding)."""
        a = TensorTerm.Tensor("a", [2, 3], True)
        out = a.poly("identity")
        inputs = {"a": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
        result = out.eval(inputs)
        # Eval pads to power-of-2; poly(identity) preserves that
        expected_padded = np.array([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0]])
        np.testing.assert_allclose(result, expected_padded)

    def test_poly_default_identity(self):
        """Poly with no func (default) is identity."""
        a = TensorTerm.Tensor("a", [2, 2], True)
        out = a.poly()
        inputs = {"a": np.array([[1.0, 2.0], [3.0, 4.0]])}
        result = out.eval(inputs)
        np.testing.assert_allclose(result, inputs["a"])

    def test_poly_polynomial_coeffs(self):
        """Poly with [c0, c1, c2, ...] computes c0 + c1*x + c2*x^2 + ..."""
        a = TensorTerm.Tensor("a", [2, 2], True)
        out = a.poly([1.0, 2.0, 0.5])  # 1 + 2*x + 0.5*x^2
        x = np.array([[0.0, 1.0], [2.0, -1.0]])
        inputs = {"a": x}
        result = out.eval(inputs)
        expected = 1.0 + 2.0 * x + 0.5 * (x**2)
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)


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
