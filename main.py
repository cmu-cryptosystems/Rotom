import random
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.heir.heir import HEIR
from backends.heir.mlir_interpreter import run_mlir_interpreter
from backends.openfhe_backend import CKKS
from backends.toy import Toy

# Import benchmarks
from benchmarks.microbenchmarks.conversion import conversion
from benchmarks.microbenchmarks.roll import roll
from benchmarks.microbenchmarks.rot_roll import rot_roll
from benchmarks.microbenchmarks.slot_bsgs_roll import slot_bsgs_roll
from benchmarks.microbenchmarks.slot_conversion import slot_conversion
from benchmarks.microbenchmarks.slot_roll import slot_roll
from benchmarks.microbenchmarks.strassens.strassens_matmul import strassens_matmul
from benchmarks.rotom_benchmarks.bert_attention import bert_attention
from benchmarks.rotom_benchmarks.convolution.convolution import convolution
from benchmarks.rotom_benchmarks.convolution.convolution_32768 import convolution_32768
from benchmarks.rotom_benchmarks.double_matmul.double_matmul_128_64_ct_ct import (
    double_matmul_128_64_ct_ct,
)
from benchmarks.rotom_benchmarks.double_matmul.double_matmul_256_128_ct_ct import (
    double_matmul_256_128_ct_ct,
)
from benchmarks.rotom_benchmarks.logreg import logreg
from benchmarks.rotom_benchmarks.matmul.matmul_128_64 import matmul_128_64
from benchmarks.rotom_benchmarks.matmul.matmul_256_128 import matmul_256_128
from benchmarks.rotom_benchmarks.ttm import ttm

# Import Rotom
from frontends.tensor import TensorTerm
from ir.dim import *
from ir.layout import *
from lower.circuit_serializer import serialize_circuit
from lower.lower import Lower
from util.checker import check_results
from wrappers.fhelipe_wrapper import FhelipeWrapper
from wrappers.viaduct_wrapper import ViaductWrapper


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
        assert kernel
        assert inputs

        circuit_ir = Lower(kernel).run()

        # Serialize circuit if requested
        if args.serialize:
            circuit_name = f"{args.microbenchmark}_{args.n}"
            output_dir = f"output/{circuit_name}"
            file_paths = serialize_circuit(circuit_ir, output_dir, circuit_name)
            print(
                f"Serialized circuit to {len(file_paths)} instruction files in {output_dir}/"
            )

        runtime = 0
        if args.backend.lower() == "toy":
            results = Toy(circuit_ir, inputs, args).run()
            check_results(kernel.term, inputs, kernel, results, runtime, args)
        elif args.backend.lower() == "ckks":
            runtime, results = CKKS(circuit_ir, inputs, args).run()
            check_results(kernel.term, inputs, kernel, results, runtime, args)
        elif args.backend.lower() == "heir":
            # HEIR backend generates MLIR output
            heir_backend = HEIR(circuit_ir, inputs, args)
            heir_backend.run()
            # Run MLIR interpreter to get results
            mlir_file = f"heir/{args.fn}/{args.fn}.mlir"
            mlir_results = run_mlir_interpreter(mlir_file)
            # Check MLIR results against kernel.term.eval()
            check_results(kernel.term, inputs, kernel, mlir_results, runtime, args)
            heir_backend.serialize_results(mlir_results)
        else:
            raise NotImplementedError("unknown backend")

        print("runtime:", runtime)
        return

    if args.benchmark:
        tensor_ir = None
        inputs = None
        n = args.n

        match args.benchmark:
            case "matmul_128_64":
                tensor_ir, inputs = matmul_128_64()
            case "matmul_256_128":
                tensor_ir, inputs = matmul_256_128()
            case "double_matmul_128_64":
                tensor_ir, inputs = double_matmul_128_64_ct_ct()
            case "double_matmul_256_128":
                tensor_ir, inputs = double_matmul_256_128_ct_ct()
            case "convolution":
                tensor_ir, inputs, n = convolution()
                args.n = n
            case "convolution_32768":
                tensor_ir, inputs, n = convolution_32768()
                args.n = n
            case "logreg":
                tensor_ir, inputs = logreg()
                args.n = n
            case "ttm":
                tensor_ir, inputs = ttm()
                args.n = n
            case "ttm_32":
                tensor_ir, inputs = ttm()
                args.n = n
            case "bert_attention":
                tensor_ir, inputs, n = bert_attention()
                args.n = n
            case "strassens":
                tensor_ir, inputs = strassens_matmul()
            case _:
                raise NotImplementedError("unknown benchmark")

        assert tensor_ir
        assert inputs
        assert n
        # Generate kernel from tensor_ir
        kernel = LayoutAssignment(tensor_ir, args).run()

        # Output found kernel
        print("found kernel:")
        for k in kernel.post_order():
            print(k)
        print()

        # Lower to circuit IR
        circuit_ir = Lower(kernel).run()

        # Serialize circuit if requested
        if args.serialize:
            circuit_name = f"{args.benchmark}_{args.n}"
            output_dir = f"output/{circuit_name}"
            file_paths = serialize_circuit(circuit_ir, output_dir, circuit_name)
            print(
                f"Serialized circuit to {len(file_paths)} instruction files in {output_dir}/"
            )

        # Run backend with result checking
        runtime = 0
        if args.backend.lower() == "toy":
            results = Toy(circuit_ir, inputs, args).run()
            check_results(tensor_ir, inputs, kernel, results, runtime, args)
        elif args.backend.lower() == "ckks":
            runtime, results = CKKS(circuit_ir, inputs, args).run()
            check_results(tensor_ir, inputs, kernel, results, runtime, args)
        elif args.backend.lower() == "heir":
            # HEIR backend generates MLIR output
            heir_backend = HEIR(circuit_ir, inputs, args)
            heir_backend.run()
            # Run MLIR interpreter to get results
            mlir_file = f"heir/{args.fn}/{args.fn}.mlir"
            mlir_results = run_mlir_interpreter(mlir_file)
            # Check MLIR results against tensor_ir.eval()
            check_results(tensor_ir, inputs, kernel, mlir_results, runtime, args)
            heir_backend.serialize_results(mlir_results)
        else:
            raise NotImplementedError("unknown backend")

        print("runtime:", runtime)
        return


def main(args):
    """Main function to run the benchmark"""

    # Check if we should run fhelipe wrapper
    if args.fhelipe:
        args.path = args.fhelipe
        w = FhelipeWrapper(args)
        comp = w.create_comp()
        # Pass args to run method so it can access not_secure flag
        results = w.run(comp, {}, args.fhelipe, args)
        return

    # Check if we should run viaduct wrapper
    if args.viaduct:
        args.path = args.viaduct
        w = ViaductWrapper(args)
        comp = w.create_comp()
        # Pass args to run method so it can access not_secure flag
        results = w.run(comp, {}, args.viaduct, args)
        return

    # Check if we should run microbenchmark or benchmark
    if args.microbenchmark != "main" or args.benchmark != "main":
        run_benchmark_or_microbenchmark(args)
        return

    # create inputs
    a = TensorTerm.Tensor("a", [64, 64], True)
    b = TensorTerm.Tensor("b", [64, 64], False)
    c = TensorTerm.Tensor("c", [64, 64], False)
    tensor_ir = a @ b @ c
    inputs = {}
    inputs["a"] = np.array(
        [[np.random.randint(0, 10) * 0.1 for j in range(64)] for i in range(64)]
    )
    inputs["b"] = np.array(
        [[np.random.randint(0, 10) * 0.1 for j in range(64)] for i in range(64)]
    )
    inputs["c"] = np.array(
        [[np.random.randint(0, 10) * 0.1 for j in range(64)] for i in range(64)]
    )

    kernel = LayoutAssignment(tensor_ir, args).run()
    for k in kernel.post_order():
        print(k)
    print()

    # lower to circuit ir
    circuit_ir = Lower(kernel).run()

    # Serialize circuit if requested
    if args.serialize:
        circuit_name = f"main_{args.serialize}_{args.n}"
        output_dir = f"output/{circuit_name}"
        file_paths = serialize_circuit(circuit_ir, output_dir, circuit_name)
        print(
            f"Serialized circuit to {len(file_paths)} instruction files in {output_dir}/"
        )

    # run backend
    runtime = 0
    if args.backend.lower() == "toy":
        results = Toy(circuit_ir, inputs, args).run()
        check_results(tensor_ir, inputs, kernel, results, runtime, args)
    elif args.backend.lower() == "ckks":
        runtime, results = CKKS(circuit_ir, inputs, args).run()
        check_results(tensor_ir, inputs, kernel, results, runtime, args)
    elif args.backend.lower() == "heir":
        # lower to HEIR MLIR
        heir_backend = HEIR(circuit_ir, inputs, args)
        heir_backend.run()
        # Run MLIR interpreter to get results
        mlir_file = f"heir/{args.fn}/{args.fn}.mlir"
        mlir_results = run_mlir_interpreter(mlir_file)
        # Check MLIR results against tensor_ir.eval()
        check_results(tensor_ir, inputs, kernel, mlir_results, runtime, args)
        heir_backend.serialize_results(mlir_results)
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
    parser.add_argument(
        "--serialize",
        type=BooleanOptionalAction,
        default=False,
        help="Serialize circuit IR to modular instruction files",
    )
    parser.add_argument("--mock", action=BooleanOptionalAction, default=False)
    parser.add_argument("--fuzz", action=BooleanOptionalAction, default=False)
    parser.add_argument("--fuzz_result", action=BooleanOptionalAction, default=False)
    parser.add_argument(
        "--conv_roll",
        action=BooleanOptionalAction,
        default=False,
        help="Use roll-based convolution for conv2d operations",
    )
    parser.add_argument(
        "--not-secure",
        action=BooleanOptionalAction,
        default=False,
        help="Disable 128-bit security level for OpenFHE backend",
    )
    parser.add_argument("--fn", type=str, default="main")
    parser.add_argument(
        "--fhelipe",
        type=str,
        default=None,
        help="Path to fhelipe benchmark directory",
    )
    parser.add_argument(
        "--viaduct",
        type=str,
        default=None,
        help="Path to viaduct benchmark file",
    )
    args = parser.parse_args()

    main(args)
