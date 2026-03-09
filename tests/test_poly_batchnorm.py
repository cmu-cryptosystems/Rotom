"""
Tests for Poly TensorTerm and BatchNorm.

Poly accepts a function (identity, silu, batchnorm, or polynomial coefficients)
and applies it element-wise. BatchNorm is implemented via Poly with
func=("batchnorm", mean_key, var_key, gamma_key, beta_key).
"""

import numpy as np

from frontends.tensor import TensorOp, TensorTerm


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


class TestBatchNorm:
    """Test BatchNorm via Poly."""

    def test_batchnorm_eval(self):
        """BatchNorm: gamma * (x - mean) / sqrt(var + eps) + beta matches reference."""
        np.random.seed(42)
        n, c = 4, 3
        x = np.random.randn(n, c).astype(np.float64)
        mean = np.random.randn(c).astype(np.float64)
        var = np.abs(np.random.randn(c).astype(np.float64)) + 0.1
        gamma = np.random.randn(c).astype(np.float64)
        beta = np.random.randn(c).astype(np.float64)
        eps = 1e-5

        # Reference
        inv_std = 1.0 / np.sqrt(var + eps)
        expected = gamma * (x - mean) * inv_std + beta

        # Rotom: build term and eval
        x_term = TensorTerm.Tensor("x", [n, c], True)
        out = TensorTerm.batchnorm(x_term, "mean", "var", "gamma", "beta", eps=eps)
        inputs = {
            "x": x,
            "mean": mean,
            "var": var,
            "gamma": gamma,
            "beta": beta,
        }
        result = out.eval(inputs)
        # Result may be padded to power-of-2; compare only the logical shape
        np.testing.assert_allclose(
            result[..., : expected.shape[-1]], expected, rtol=1e-9, atol=1e-9
        )

    def test_batchnorm_after_matmul(self):
        """BatchNorm after a matmul: (x @ w).poly(batchnorm) then eval."""
        np.random.seed(43)
        batch, in_dim, out_dim = 2, 4, 3
        x = np.random.randn(batch, in_dim).astype(np.float64)
        w = np.random.randn(in_dim, out_dim).astype(np.float64)
        mean = np.zeros(out_dim)
        var = np.ones(out_dim) * 0.5
        gamma = np.ones(out_dim)
        beta = np.zeros(out_dim)

        x_term = TensorTerm.Tensor("x", [batch, in_dim], True)
        w_term = TensorTerm.Tensor("w", [in_dim, out_dim], False)
        linear = x_term @ w_term
        out = TensorTerm.batchnorm(linear, "mean", "var", "gamma", "beta")

        inputs = {
            "x": x,
            "w": w,
            "mean": mean,
            "var": var,
            "gamma": gamma,
            "beta": beta,
        }
        result = out.eval(inputs)

        raw = x @ w
        inv_std = 1.0 / np.sqrt(var + 1e-5)
        expected = gamma * (raw - mean) * inv_std + beta
        np.testing.assert_allclose(
            result[..., : expected.shape[-1]], expected, rtol=1e-9, atol=1e-9
        )


class TestPolyCallable:
    """Test Poly with a custom Python function (eval-only)."""

    def test_poly_callable_two_batchnorms(self):
        """Two BatchNorms with different params via callables (no key naming)."""
        np.random.seed(44)
        n, c = 4, 3
        x = np.random.randn(n, c).astype(np.float64)
        # First BN params
        m1 = np.random.randn(c)
        v1 = np.abs(np.random.randn(c)) + 0.1
        g1 = np.ones(c)
        b1 = np.zeros(c)
        # Second BN params (different)
        m2 = np.random.randn(c)
        v2 = np.abs(np.random.randn(c)) + 0.1
        g2 = np.ones(c) * 2
        b2 = np.ones(c)

        def bn(x, m, v, g, b, eps=1e-5):
            return g * (x - m) / np.sqrt(v + eps) + b

        def extend_params(m, v, g, b, last_dim):
            if last_dim <= len(m):
                return m, v, g, b
            pad = last_dim - len(m)
            return (
                np.concatenate([m, np.zeros(pad)]),
                np.concatenate([v, np.ones(pad)]),
                np.concatenate([g, np.ones(pad)]),
                np.concatenate([b, np.zeros(pad)]),
            )

        x_term = TensorTerm.Tensor("x", [n, c], True)
        # First BN as callable (closure over m1, v1, g1, b1)
        out1 = x_term.poly(lambda t: bn(t, *extend_params(m1, v1, g1, b1, t.shape[-1])))
        # Second BN as callable (closure over m2, v2, g2, b2)
        out2 = x_term.poly(lambda t: bn(t, *extend_params(m2, v2, g2, b2, t.shape[-1])))

        inputs = {"x": x}
        r1 = out1.eval(inputs)
        r2 = out2.eval(inputs)

        expected1 = bn(x, m1, v1, g1, b1)
        expected2 = bn(x, m2, v2, g2, b2)
        np.testing.assert_allclose(r1[..., :c], expected1, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(r2[..., :c], expected2, rtol=1e-9, atol=1e-9)
        # Results differ because params differ
        assert not np.allclose(r1[..., :c], r2[..., :c])


class TestPolyShapeAnalysis:
    """Test that shape analysis handles POLY."""

    def test_poly_shape_preserved(self):
        """Shape analysis: POLY preserves input shape."""
        from ir.analysis.shape import Shape

        a = TensorTerm(TensorOp.TENSOR, ["a", [5, 7], True])
        poly_term = TensorTerm(TensorOp.POLY, [a, "silu"])
        shape_analyzer = Shape(poly_term)
        shape_analyzer.run()
        assert shape_analyzer.get_shape(poly_term) == [5, 7]
        assert shape_analyzer.get_padded_shape(poly_term) == [8, 8]
