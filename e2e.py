from argparse import ArgumentParser, BooleanOptionalAction

from assignment.assignment import LayoutAssignment
from util.benchmark_layout_plan_cache import (
    clear_benchmark_layout_plan_cache_override,
    maybe_install_layout_plan_cache_from_args,
)
from backends.heir.heir import HEIR
from backends.heir.mlir_interpreter import run_mlir_interpreter
from backends.toy import Toy

# heir benchmarks
from benchmarks.e2e.mnist.mnist_poly import mnist_poly
from benchmarks.e2e.mnist.mnist_poly_call import mnist_poly_call

# Import Rotom
from ir.dim import *
from ir.layout import *
from lower.lower import Lower
from util.checker import check_label, check_results
from util.input_serializer import serialize_mlp_mnist_inputs


def main(args):
    """Run either a microbenchmark or benchmark based on args"""
    try:
        tensor_ir = None
        inputs = None
        n = args.n

        match args.fn:
            case "mnist_poly_call":
                tensor_ir, inputs, label = mnist_poly_call(args.label)
                args.n = n
            case "mnist_poly":
                tensor_ir, inputs, label = mnist_poly(args.label)
                args.n = n
            case "":
                raise NotImplementedError("benchmark not set")
            case _:
                raise NotImplementedError(f"unknown benchmark: {args.fn}")

        assert tensor_ir
        assert inputs
        assert n

        maybe_install_layout_plan_cache_from_args(args, args.fn)

        kernel = LayoutAssignment(tensor_ir, args).run()

        print("found kernel:")
        for k in kernel.post_order():
            print(k)
        print()

        circuit_ir = Lower(kernel).run()

        runtime = 0
        backend_results = None
        if args.backend.lower() == "toy":
            backend_results = Toy(circuit_ir, inputs, args).run()
            check_results(tensor_ir, inputs, kernel, backend_results, runtime, args)
            check_label(kernel, backend_results, label)
        elif args.backend.lower() == "heir":
            heir_backend = HEIR(circuit_ir, inputs, args)
            heir_backend.run()
            mlir_file = f"heir/{args.fn}/{args.fn}.mlir"
            mlir_results = run_mlir_interpreter(mlir_file, n)
            check_results(tensor_ir, inputs, kernel, mlir_results, runtime, args)
            check_label(kernel, mlir_results, label)
            heir_backend.serialize_results(mlir_results)
            backend_results = mlir_results
        else:
            raise NotImplementedError("unknown backend")

        if args.serialize_inputs:
            match args.fn:
                case "mnist":
                    serialize_mlp_mnist_inputs(kernel)
                case _:
                    pass

        return backend_results, label
    finally:
        clear_benchmark_layout_plan_cache_override()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fn", type=str, default="")
    parser.add_argument("--benchmark", type=str, default="main")
    parser.add_argument("--backend", default="toy")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--rolls", action=BooleanOptionalAction, default=False)
    parser.add_argument("--net", default="lan")
    parser.add_argument("--cache", action=BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cache-layout-plans",
        action=BooleanOptionalAction,
        default=True,
        help="Persist apply_layout plan pickles under .cache/layout_plans/<fn>/n_<n>/",
    )
    parser.add_argument(
        "--not-secure",
        action=BooleanOptionalAction,
        default=False,
        help="Disable 128-bit security level for OpenFHE backend",
    )
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument(
        "--serialize_inputs", action=BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    main(args)
