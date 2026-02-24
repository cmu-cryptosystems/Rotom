#!/usr/bin/env python3
"""
Compile all Rotom benchmarks without running them.
This script performs layout assignment, optimization, and lowering to circuit IR
but does not execute the backend.
"""

import time
from argparse import Namespace

import numpy as np

from assignment.assignment import LayoutAssignment
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
from benchmarks.rotom_benchmarks.mlp_mnist import mlp_mnist
from benchmarks.rotom_benchmarks.matmul.matmul_128_64 import matmul_128_64
from benchmarks.rotom_benchmarks.matmul.matmul_256_128 import matmul_256_128
from benchmarks.rotom_benchmarks.ttm import ttm
from lower.lower import Lower
from opt.pack_tensor_lifting import run_pack_tensor_lifting


def compile_benchmark(benchmark_name, tensor_ir, inputs, n, args):
    """Compile a benchmark: layout assignment, optimization, and lowering."""
    print(f"\n{'='*60}")
    print(f"Compiling: {benchmark_name}")
    print(f"{'='*60}")

    # Update args with benchmark-specific n value
    args.n = n

    # Start compile time measurement
    compile_start = time.time()

    try:
        # Generate kernel from tensor_ir
        kernel = LayoutAssignment(tensor_ir, args).run()

        # Apply pack tensor lifting optimization
        kernel = run_pack_tensor_lifting(
            kernel, args.net if hasattr(args, "net") else "lan"
        )

        # Output found kernel
        print("Found kernel:")
        for k in kernel.post_order():
            print(f"  {k}")

        # Lower to circuit IR
        circuit_ir = Lower(kernel).run()

        # End compile time measurement
        compile_time = time.time() - compile_start
        print(f"\n✓ Compilation successful!")
        print(f"  Compile time: {compile_time:.4f} seconds")
        print(f"  Slots (n): {n}")

        return True, compile_time

    except Exception as e:
        compile_time = time.time() - compile_start
        print(f"\n✗ Compilation failed!")
        print(f"  Error: {str(e)}")
        print(f"  Time before error: {compile_time:.4f} seconds")
        return False, compile_time


def main():
    """Compile all Rotom benchmarks."""

    # Default arguments for compilation
    base_args = Namespace(
        n=4096,
        rolls=True,
        strassens=False,
        net="lan",
        cache=False,
        mock=False,
        not_secure=False,
    )

    # Define all benchmarks with their parameters
    benchmarks = [
        # Rotom benchmarks
        {
            "name": "matmul_128_64",
            "n": 8192,
            "func": matmul_128_64,
            "not_secure": True,
        },
        {
            "name": "matmul_256_128",
            "n": 32768,
            "func": matmul_256_128,
            "not_secure": False,
        },
        {
            "name": "double_matmul_128_64",
            "n": 8192,
            "func": double_matmul_128_64_ct_ct,
            "not_secure": True,
        },
        {
            "name": "double_matmul_256_128",
            "n": 32768,
            "func": double_matmul_256_128_ct_ct,
            "not_secure": False,
        },
        {
            "name": "convolution",
            "n": None,  # Will be set by the function
            "func": convolution,
            "not_secure": True,
        },
        {
            "name": "convolution_32768",
            "n": None,  # Will be set by the function
            "func": convolution_32768,
            "not_secure": False,
        },
        {
            "name": "logreg",
            "n": None,  # Will be set by the function
            "func": logreg,
            "not_secure": False,
        },
        {
            "name": "mlp_mnist",
            "n": 8192,
            "func": mlp_mnist,
            "not_secure": True,
        },
        {
            "name": "ttm",
            "n": None,  # Will be set by the function
            "func": ttm,
            "not_secure": False,
        },
        {
            "name": "bert_attention",
            "n": 8192,
            "func": bert_attention,
            "not_secure": False,
        },
        {
            "name": "strassens",
            "n": 8192,
            "func": strassens_matmul,
            "not_secure": True,
        },
    ]

    # Microbenchmarks (with size parameter)
    microbenchmarks = [
        {
            "name": "conversion",
            "n": 8192,
            "size": 4,
            "func": conversion,
            "not_secure": True,
        },
        {
            "name": "roll",
            "n": 8192,
            "size": 4,
            "func": roll,
            "not_secure": True,
        },
        {
            "name": "rot_roll",
            "n": 8192,
            "size": 4,
            "func": rot_roll,
            "not_secure": True,
        },
        {
            "name": "slot_conversion",
            "n": 8192,
            "size": 4,
            "func": slot_conversion,
            "not_secure": True,
        },
        {
            "name": "slot_roll",
            "n": 8192,
            "size": 4,
            "func": slot_roll,
            "not_secure": True,
        },
        {
            "name": "slot_bsgs_roll",
            "n": 8192,
            "size": 4,
            "func": slot_bsgs_roll,
            "not_secure": True,
        },
    ]

    results = []
    total_start = time.time()

    print("=" * 60)
    print("Rotom Benchmark Compilation Test Suite")
    print("=" * 60)
    print(f"Total benchmarks: {len(benchmarks) + len(microbenchmarks)}")
    print()

    # Compile regular benchmarks
    for bench in benchmarks:
        args = Namespace(**vars(base_args))
        args.not_secure = bench.get("not_secure", False)

        try:
            # Call benchmark function
            if bench["n"] is None:
                # Some benchmarks set n themselves
                tensor_ir, inputs, n = bench["func"]()
            else:
                tensor_ir, inputs = bench["func"]()
                n = bench["n"]

            success, compile_time = compile_benchmark(
                bench["name"], tensor_ir, inputs, n, args
            )
            results.append((bench["name"], success, compile_time))

        except Exception as e:
            print(f"\n✗ Failed to load benchmark '{bench['name']}': {str(e)}")
            results.append((bench["name"], False, 0.0))

    # Compile microbenchmarks
    for bench in microbenchmarks:
        args = Namespace(**vars(base_args))
        args.not_secure = bench.get("not_secure", False)
        args.size = bench.get("size", 4)

        try:
            # Call microbenchmark function
            kernel, inputs = bench["func"](bench["n"], bench["size"])
            n = bench["n"]

            # For microbenchmarks, we already have a kernel, so we just need to lower it
            print(f"\n{'='*60}")
            print(f"Compiling: {bench['name']}")
            print(f"{'='*60}")

            compile_start = time.time()
            try:
                circuit_ir = Lower(kernel).run()
                compile_time = time.time() - compile_start
                print(f"\n✓ Compilation successful!")
                print(f"  Compile time: {compile_time:.4f} seconds")
                print(f"  Slots (n): {n}, Size: {bench['size']}")
                results.append((bench["name"], True, compile_time))
            except Exception as e:
                compile_time = time.time() - compile_start
                print(f"\n✗ Compilation failed!")
                print(f"  Error: {str(e)}")
                results.append((bench["name"], False, compile_time))

        except Exception as e:
            print(f"\n✗ Failed to load microbenchmark '{bench['name']}': {str(e)}")
            results.append((bench["name"], False, 0.0))

    # Print summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("Compilation Summary")
    print("=" * 60)

    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    total_compile_time = sum(ct for _, _, ct in results)

    print(f"Total benchmarks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total compilation time: {total_compile_time:.4f} seconds")
    print(f"Total script time: {total_time:.4f} seconds")
    print()

    print("Detailed Results:")
    print("-" * 60)
    for name, success, compile_time in results:
        status = "✓" if success else "✗"
        print(f"{status} {name:30s} {compile_time:8.4f}s")

    if failed > 0:
        print("\n" + "=" * 60)
        print("Failed Benchmarks:")
        print("=" * 60)
        for name, success, _ in results:
            if not success:
                print(f"  ✗ {name}")

    print("\n" + "=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
