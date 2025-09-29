import random
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.openfhe_backend import CKKS
from backends.toy import Toy

# Import microbenchmark functions
from benchmarks.microbenchmarks.conversion import conversion
from benchmarks.microbenchmarks.roll import roll
from benchmarks.microbenchmarks.rot_roll import rot_roll
from benchmarks.microbenchmarks.slot_bsgs_roll import slot_bsgs_roll
from benchmarks.microbenchmarks.slot_conversion import slot_conversion
from benchmarks.microbenchmarks.slot_roll import slot_roll
from benchmarks.microbenchmarks.ttm_micro import ttm_micro
from benchmarks.microbenchmarks.ttm_micro_32 import ttm_micro_32
from benchmarks.rotom_benchmarks.bert_attention import bert_attention
from benchmarks.rotom_benchmarks.convolution import convolution
from benchmarks.rotom_benchmarks.convolution_32768 import convolution_32768

# Import benchmark functions
from benchmarks.rotom_benchmarks.distance import distance
from benchmarks.rotom_benchmarks.double_matmul.double_matmul_128_64 import (
    double_matmul_128_64,
)
from benchmarks.rotom_benchmarks.double_matmul.double_matmul_256_128 import (
    double_matmul_256_128,
)
from benchmarks.rotom_benchmarks.logreg import logreg
from benchmarks.rotom_benchmarks.retrieval import retrieval
from benchmarks.rotom_benchmarks.ttm import ttm
from frontends.tensor import TensorTerm
from ir.dim import *
from ir.kernel_cost import KernelCost
from ir.layout import *
from lower.lower import Lower
from util.checker import check_results


def run_benchmark_or_microbenchmark(args):
    """Run either a microbenchmark or benchmark based on args"""

    if args.microbenchmark != "main":
        kernel = None
        n = args.n

        match args.microbenchmark:
            case "conversion":
                kernel, inputs = conversion(args.n, args.size)
            case "roll":
                kernel, inputs = roll(args.n, args.size)
            case "rot_roll":
                kernel, inputs = rot_roll(args.n, args.size)
            case "slot_conversion":
                kernel, inputs = slot_conversion(args.n, args.size)
            case "slot_roll":
                kernel, inputs = slot_roll(args.n, args.size)
            case "slot_bsgs_roll":
                kernel, inputs = slot_bsgs_roll(args.n, args.size)
            case "ttm":
                kernel, inputs = ttm_micro(args.n)
            case "ttm_32":
                kernel, inputs = ttm_micro_32(args.n)

        assert kernel
        assert inputs

        print(
            "picked kernel:",
            kernel,
            KernelCost(kernel, args.net).total_cost(),
            KernelCost(kernel, args.net).comm_cost(),
        )
        print(KernelCost(kernel, args.net).total_operations())
        print(kernel.layout.term)
        print()
        for k in kernel.post_order():
            print(k)
            print(k, KernelCost(k, args.net).op_cost())
            print(KernelCost(k, args.net).ops())
        print()

        circuit_ir = Lower(kernel).run()

        runtime = 0
        if args.backend.lower() == "toy":
            results = Toy(circuit_ir, inputs, args).run()
        elif args.backend.lower() == "ckks":
            runtime, results = CKKS(circuit_ir, inputs, args).run()
        elif args.backend.lower() == "heir":
            # Note: HEIR backend not implemented yet
            raise NotImplementedError("HEIR backend not implemented")
        elif args.backend.lower() == "heco":
            # Note: HECO backend not implemented yet
            raise NotImplementedError("HECO backend not implemented")
        else:
            raise NotImplementedError("unknown backend")

        print("runtime:", runtime)
        return

    if args.benchmark:
        tensor_ir = None
        inputs = None
        n = args.n

        match args.benchmark:
            case "distance":
                tensor_ir, inputs = distance()
            case "ttm":
                tensor_ir, inputs = ttm()
            case "retrieval":
                tensor_ir, inputs = retrieval()
            case "double_matmul_128_64":
                tensor_ir, inputs = double_matmul_128_64()
            case "double_matmul_256_128":
                tensor_ir, inputs = double_matmul_256_128()
            case "logreg":
                tensor_ir, inputs = logreg()
            case "convolution":
                tensor_ir, inputs = convolution()
            case "convolution_32768":
                tensor_ir, inputs = convolution_32768()
            case "bert_attention":
                tensor_ir, inputs = bert_attention()
            case _:
                raise NotImplementedError("unknown benchmark")

        assert tensor_ir
        assert inputs
        assert n
    else:
        args.benchmark = "main"

        n = 8
        a = TensorTerm.Tensor("a", [4, 2], True)
        b = TensorTerm.Tensor("b", [2, 4], False)
        c = TensorTerm.Tensor("c", [4, 2], False)
        tensor_ir = a @ b @ c
        inputs = {}
        inputs["a"] = np.array([[i * 4 + j for j in range(2)] for i in range(4)])
        inputs["b"] = np.array([[i * 2 + j for j in range(4)] for i in range(2)])
        inputs["c"] = np.array([[i * 4 + j for j in range(2)] for i in range(4)])


def main(args):
    """Main function to run the benchmark"""

    # Check if we should run microbenchmark or benchmark
    if args.microbenchmark != "main" or args.benchmark != "main":
        run_benchmark_or_microbenchmark(args)
        return

    # Original main logic for default case
    # generate inputs
    n = args.n

    # create inputs
    inputs = {}
    inputs["a"] = np.array(
        [[random.randint(0, 2) for i in range(64)] for j in range(64)]
    )
    inputs["b"] = np.array([random.randint(0, 2) for i in range(64)])

    # generate test case
    a = TensorTerm.Tensor("a", [64, 64], True)
    b = TensorTerm.Tensor("b", [64], False)
    tensor_ir = a @ b
    kernel = LayoutAssignment(tensor_ir, args).run()

    for k in kernel.post_order():
        print(k)
    print()

    # lower to circuit ir
    circuit_ir = Lower(kernel).run()

    # run backend
    runtime = 0
    if args.backend.lower() == "toy":
        results = Toy(circuit_ir, inputs, args).run()
        check_results(tensor_ir, inputs, kernel, results, runtime, args)
    elif args.backend.lower() == "ckks":
        runtime, results = CKKS(circuit_ir, inputs, args).run()
        check_results(tensor_ir, inputs, kernel, results, runtime, args)
    else:
        raise NotImplementedError("unknown backend")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--benchmark", default="main")
    parser.add_argument("--microbenchmark", default="main")
    parser.add_argument("--backend", default="toy")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--rolls", action=BooleanOptionalAction, default=False)
    parser.add_argument("--strassens", action=BooleanOptionalAction, default=False)
    parser.add_argument("--net", default="lan")
    parser.add_argument("--cache", action=BooleanOptionalAction, default=False)
    parser.add_argument("--serialize", action=BooleanOptionalAction, default=False)
    parser.add_argument("--mock", action=BooleanOptionalAction, default=False)
    parser.add_argument("--fuzz", action=BooleanOptionalAction, default=False)
    parser.add_argument("--fuzz_result", action=BooleanOptionalAction, default=False)
    parser.add_argument(
        "--not-secure",
        action=BooleanOptionalAction,
        default=False,
        help="Disable 128-bit security level for OpenFHE backend",
    )
    parser.add_argument("--fn", type=str, default="fn")
    args = parser.parse_args()

    main(args)
