"""
Small-scale unit test for a BERT-style attention block.

This test mirrors the structure of ``benchmarks/rotom_benchmarks/bert_attention.py``
but with tiny tensor sizes so it is easier to debug end-to-end through the
Rotom pipeline (TensorTerm -> LayoutAssignment -> Lower -> backend).
"""

import random

import numpy as np

from assignment.assignment import LayoutAssignment
from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.conftest import assert_results_equal, run_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestBertAttentionSmall:
    """Small BERT-style attention test using tiny dimensions."""

    def _create_bert_attention_small_computation(self, inputs):
        """
        Build a tiny attention computation using TensorTerm operations.

        Shapes (all very small for debugging):
            - h  : [seq_len, hidden_dim]      = [4, 8] (ciphertext)
            - wq : [hidden_dim, hidden_dim]   = [8, 8] (plaintext)
            - wk : [hidden_dim, hidden_dim]   = [8, 8] (plaintext)
            - wv : [hidden_dim, hidden_dim]   = [8, 8] (plaintext)
            - bq, bk, bv : [hidden_dim]       = [8]    (plaintext)

        Multi-head-style blocking:
            - num_heads = 2, head_dim = 4, so hidden_dim = num_heads * head_dim.
            - We follow the same reshape/permute pattern as the full benchmark.
        """
        seq_len = 4
        hidden_dim = 8
        num_heads = 2
        head_dim = hidden_dim // num_heads

        # Define TensorTerms matching the numpy input shapes.
        h = TensorTerm.Tensor("h", [seq_len, hidden_dim], True)
        wq = TensorTerm.Tensor("wq", [hidden_dim, hidden_dim], False)
        bq = TensorTerm.Tensor("bq", [hidden_dim], False)
        wk = TensorTerm.Tensor("wk", [hidden_dim, hidden_dim], False)
        bk = TensorTerm.Tensor("bk", [hidden_dim], False)
        wv = TensorTerm.Tensor("wv", [hidden_dim, hidden_dim], False)
        bv = TensorTerm.Tensor("bv", [hidden_dim], False)

        # Linear projections
        q = h @ wq + bq
        k = h @ wk + bk
        v = h @ wv + bv

        # Reshape / permute into [num_heads, seq_len, head_dim] as in the benchmark.
        blocked_q = q.reshape(1, {1: num_heads, 2: head_dim}).permute(
            {0: 1, 1: 0, 2: 2}
        )
        # permute transposes k per head
        blocked_kt = k.reshape(1, {1: num_heads, 2: head_dim}).permute(
            {0: 2, 1: 0, 2: 1}
        )
        blocked_v = v.reshape(1, {1: num_heads, 2: head_dim}).permute(
            {0: 1, 1: 0, 2: 2}
        )

        head_results = None
        for h_idx in range(num_heads):
            # After permute, each slice is [seq_len, head_dim].
            q_h = blocked_q[h_idx, :, :]
            k_h = blocked_kt[h_idx, :, :]
            v_h = blocked_v[h_idx, :, :]

            # Simple (unnormalized) attention: (Q K^T) V
            qk_h = q_h @ k_h
            out_h = qk_h @ v_h

            if head_results is None:
                head_results = out_h
            else:
                head_results = head_results + out_h

        return head_results

    def _run_test_case(self, tensor_ir, inputs, args, backend):
        """Helper to run the tiny attention computation end-to-end."""
        # Generate expected result using the frontend evaluator (includes padding semantics).
        expected = tensor_ir.eval(inputs)

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()

        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)

        # Apply layout to expected result and compare.
        expected_cts = apply_layout(expected, kernel.layout)
        assert_results_equal(expected_cts, results, backend)

    def test_bert_attention_small(self, backend):
        """
        End-to-end test of a tiny BERT-style attention block with random binary inputs.
        """
        # Create args with a small ring dimension suitable for debugging.
        args = get_default_args()
        args.n = 32
        args.rolls = True
        args.benchmark = "bert_attention_small"

        # Small shapes for easier inspection.
        seq_len = 4
        hidden_dim = 8

        # Create random binary inputs similar to the benchmark style.
        inputs = {}
        inputs["h"] = np.array(
            [
                [random.choice(range(2)) for _ in range(hidden_dim)]
                for _ in range(seq_len)
            ]
        )
        inputs["wq"] = np.array(
            [
                [random.choice(range(2)) for _ in range(hidden_dim)]
                for _ in range(hidden_dim)
            ]
        )
        inputs["bq"] = np.array([random.choice(range(2)) for _ in range(hidden_dim)])
        inputs["wk"] = np.array(
            [
                [random.choice(range(2)) for _ in range(hidden_dim)]
                for _ in range(hidden_dim)
            ]
        )
        inputs["bk"] = np.array([random.choice(range(2)) for _ in range(hidden_dim)])
        inputs["wv"] = np.array(
            [
                [random.choice(range(2)) for _ in range(hidden_dim)]
                for _ in range(hidden_dim)
            ]
        )
        inputs["bv"] = np.array([random.choice(range(2)) for _ in range(hidden_dim)])

        tensor_ir = self._create_bert_attention_small_computation(inputs)
        self._run_test_case(tensor_ir, inputs, args, backend)
