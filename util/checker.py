from ir.kernel_cost import KernelCost
from util.layout_util import apply_layout


def check_results(tensor_ir, inputs, kernel, results, runtime, args):
    """Check results of the benchmark"""
    expected_cts = apply_layout(tensor_ir.eval(inputs), kernel.layout)
    if args.benchmark == "main" and expected_cts != results:
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

    assert expected_cts == results
    print("passed!")
