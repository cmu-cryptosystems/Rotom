import numpy as np

from ir.kernel_cost import KernelCost
from util.layout_util import apply_layout


def check_results(tensor_ir, inputs, kernel, results, runtime, args):
    """Check results of the benchmark"""
    expected_cts = apply_layout(tensor_ir.eval(inputs), kernel.layout)

    # Check if values are close instead of exact equality
    all_close = True
    max_diff = 0.0

    for expected, result in zip(expected_cts, results):
        if not np.allclose(expected, result, rtol=1e-2, atol=1e-2):
            all_close = False
            diff = np.array(expected) - np.array(result)
            max_diff = max(max_diff, np.max(np.abs(diff)))

    if args.benchmark == "main" and not all_close:
        print("expected:")
        for expected in expected_cts:
            print(expected)
        print()

        if len(expected_cts) > len(results):
            expected_cts = expected_cts[: len(results)]

        print("result:")
        for result in results:
            print(result)
        print()

        print("diff:")
        for expected, result in zip(expected_cts, results):
            print([e - r for e, r in zip(expected, result)])
        print()
        print("runtime:", runtime + KernelCost(kernel, args.net).real_comm_cost())
        print(kernel.layout)

    assert all_close, f"Values not close enough. Max diff: {max_diff}"
    print("passed!")
