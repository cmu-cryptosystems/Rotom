from argparse import ArgumentParser, BooleanOptionalAction

from assignment.assignment import LayoutAssignment
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
            raise NotImplementedError(f"benchmark not set")
        case _:
            raise NotImplementedError(f"unknown benchmark: {args.fn}")

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

    # Run backend with result checking
    runtime = 0
    backend_results = None
    if args.backend.lower() == "toy":
        backend_results = Toy(circuit_ir, inputs, args).run()
        check_results(tensor_ir, inputs, kernel, backend_results, runtime, args)
        check_label(kernel, backend_results, label)
    elif args.backend.lower() == "heir":
        # HEIR backend generates MLIR output
        heir_backend = HEIR(circuit_ir, inputs, args)
        heir_backend.run()
        # Run MLIR interpreter to get results
        mlir_file = f"heir/{args.fn}/{args.fn}.mlir"
        mlir_results = run_mlir_interpreter(mlir_file, n)
        # Check MLIR results against tensor_ir.eval()
        check_results(tensor_ir, inputs, kernel, mlir_results, runtime, args)
        check_label(kernel, backend_results, label)

        # Serialize results
        heir_backend.serialize_results(mlir_results)
        backend_results = mlir_results
    else:
        raise NotImplementedError("unknown backend")

    if args.serialize_inputs:
        match args.fn:
            case "mnist":
                serialize_mlp_mnist_inputs(kernel)
            case _:
                raise NotImplementedError(f"unknown benchmark: {args.fn}")

    return backend_results, label


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
