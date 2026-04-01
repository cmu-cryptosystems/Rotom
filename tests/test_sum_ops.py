import numpy as np
import torch

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def _run_and_compare(tensor_term: TensorTerm, inputs: dict, *, n: int) -> None:
    args = get_default_args()
    args.backend = "toy"
    args.n = n
    args.rolls = False
    # Keep the unit test focused on correctness + shape compatibility.
    args.skip_toy_eval_checks = True

    kernel = LayoutAssignment(tensor_term, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()

    dense_eval = tensor_term.eval(inputs)
    expected_cts = apply_layout(dense_eval, kernel.layout)

    assert len(expected_cts) == len(backend_results)
    for exp, res in zip(expected_cts, backend_results):
        exp_a = np.asarray(exp, dtype=np.float64).ravel()
        res_a = np.asarray(res, dtype=np.float64).ravel()
        assert exp_a.shape == res_a.shape
        assert np.isfinite(res_a).all(), "Toy output should be finite"


def test_sequential_sum_dim1_then_dim1_shape_and_lowering() -> None:
    """
    Corner case: two consecutive SUM ops that both reduce `dim_idx=1`.

    Starting tensor is [C, H, W] and reducing SUM(1) twice should produce shape [C].
    """
    torch.manual_seed(0)
    np.random.seed(0)

    x = np.random.randn(4, 8, 8).astype(np.float64)
    inputs = {"x": x}

    t = TensorTerm.Tensor("x", [4, 8, 8], True)
    t = t.sum(1).sum(1)  # [4, 8, 8] -> [4, 8] -> [4]

    # Use a small n to increase chance of gap/tiling paths.
    _run_and_compare(t, inputs, n=64)


def test_sequential_sum_dim0_then_dim1_shape_and_lowering() -> None:
    """
    Another renumbering corner: SUM(0) then SUM(1) after rank shrink.
    """
    torch.manual_seed(0)
    np.random.seed(1)

    x = np.random.randn(4, 8, 8).astype(np.float64)
    inputs = {"x": x}

    t = TensorTerm.Tensor("x", [4, 8, 8], True)
    t = t.sum(0).sum(1)  # [4, 8, 8] -> [8, 8] -> [8]

    _run_and_compare(t, inputs, n=64)
