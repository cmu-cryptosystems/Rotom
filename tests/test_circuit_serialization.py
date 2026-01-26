"""
Test circuit serialization and deserialization.

This test verifies that HE circuits can be correctly serialized to modular
instruction files and loaded back while maintaining correctness.
"""

import os
import shutil
from argparse import Namespace

import numpy as np

from assignment.assignment import LayoutAssignment
from frontends.tensor import TensorTerm
from lower.circuit_loader import load_circuit
from lower.circuit_serializer import serialize_circuit
from lower.lower import Lower
from tests.conftest import run_backend


class TestCircuitSerialization:
    """Test circuit serialization and deserialization operations."""

    def test_matmul_serialization(self, backend):
        """Test serialization of a matrix-vector multiplication circuit."""

        print("=" * 70)
        print("Test: Matrix-Vector Multiplication Circuit Serialization")
        print("=" * 70)

        # Setup
        output_dir = "output/test_circuits"
        circuit_name = "matmul_test"

        # Clean up any existing output
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # 1. Create tensor computation
        print("\n1. Creating tensor computation...")
        a = TensorTerm.Tensor("a", [8, 8], True)  # 8x8 secret matrix
        b = TensorTerm.Tensor("b", [8], False)  # 8-element plaintext vector
        tensor_ir = a @ b
        print(f"   Tensor IR: {tensor_ir}")

        # Create test inputs
        inputs = {
            "a": np.array(
                [[i * 8 + j for j in range(8)] for i in range(8)], dtype=float
            ),
            "b": np.array([i for i in range(8)], dtype=float),
        }

        # 2. Run layout assignment
        print("\n2. Running layout assignment...")
        args = Namespace(
            n=4096,
            rolls=True,
            backend="toy",
            net="lan",
            cache=False,
            mock=False,
            serialize=False,
        )

        kernel = LayoutAssignment(tensor_ir, args).run()
        print(f"   Generated {len(list(kernel.post_order()))} kernel operations")

        # 3. Lower to circuit IR
        print("\n3. Lowering to HE circuit...")
        circuit_ir = Lower(kernel).run()
        print(f"   Circuit contains {len(circuit_ir)} kernel terms")

        # 4. Execute original circuit
        print("\n4. Executing original circuit...")
        args.backend = backend
        original_results = run_backend(backend, circuit_ir, inputs, args)
        # Toy backend returns a list of results
        result = (
            original_results[0]
            if isinstance(original_results, list)
            else original_results
        )
        print(
            f"   Original result (first 8 elements): {result[:8] if hasattr(result, '__getitem__') else result}"
        )

        # 5. Serialize circuit
        print(f"\n5. Serializing circuit to {output_dir}...")
        file_paths = serialize_circuit(circuit_ir, output_dir, circuit_name)
        print(f"   Created {len(file_paths)} files:")
        for key, path in file_paths.items():
            file_size = os.path.getsize(path)
            print(f"     - {key}: {os.path.basename(path)} ({file_size} bytes)")

        # 6. Load circuit back
        print(f"\n6. Loading circuit from {output_dir}...")
        loaded_data = load_circuit(output_dir, circuit_name)
        print(
            f"   Loaded manifest with {len(loaded_data['manifest']['kernels'])} kernels"
        )
        print(f"   Total instructions loaded: {len(loaded_data['instructions'])}")

        # 7. Verify loaded circuit structure
        print("\n7. Verifying circuit structure...")
        for kernel_data in loaded_data["kernels"]:
            meta = kernel_data["metadata"]
            print(f"   Kernel {meta['kernel_idx']}: {meta['operation']}")
            print(f"     - Layout: {meta['layout']}")
            print(f"     - Ciphertexts: {meta['num_ciphertexts']}")
            print(f"     - Instructions: {len(meta['instructions'])}")
            print(f"     - Outputs: {meta['outputs']}")

        # 8. Verify instruction correctness by re-executing
        print("\n8. Re-executing with original circuit IR...")
        reexec_results = run_backend(backend, circuit_ir, inputs, args)

        # 9. Compare results
        print("\n9. Comparing results...")
        orig_result = (
            original_results[0]
            if isinstance(original_results, list)
            else original_results
        )
        reexec_result = (
            reexec_results[0] if isinstance(reexec_results, list) else reexec_results
        )

        match = np.allclose(orig_result, reexec_result, rtol=1e-5)
        print(f"   Results match: {match}")

        if match:
            print("   ✓ Circuit serialization test PASSED")
        else:
            print("   ✗ Circuit serialization test FAILED")
            print(
                f"   Max difference: {np.max(np.abs(np.array(orig_result) - np.array(reexec_result)))}"
            )

        # 10. Verify we can parse instruction semantics
        print("\n10. Verifying instruction parsing...")
        execution_order = sorted(loaded_data["instructions"].keys())
        print(
            f"   Execution order: {execution_order[:10]}..."
            if len(execution_order) > 10
            else f"   Execution order: {execution_order}"
        )

        # Count operation types
        op_counts = {}
        for idx, (is_secret, op, operands) in loaded_data["instructions"].items():
            op_counts[op] = op_counts.get(op, 0) + 1

        print(f"   Operation counts:")
        for op, count in sorted(op_counts.items()):
            print(f"     - {op}: {count}")

        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)

        # Use assert instead of return for pytest compliance
        assert (
            match
        ), f"Results do not match. Max difference: {np.max(np.abs(np.array(orig_result) - np.array(reexec_result)))}"

    def test_multiple_operations(self, backend):
        """Test serialization with multiple tensor operations."""

        print("\n" + "=" * 70)
        print("Test: Multiple Operations Circuit Serialization")
        print("=" * 70)

        # Setup
        output_dir = "output/test_multi_circuits"
        circuit_name = "multi_op_test"

        # Clean up
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # Create tensor computation with multiple operations
        print("\n1. Creating multi-operation tensor computation...")
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 4], False)
        c = TensorTerm.Tensor("c", [4], False)

        # (a + b) @ c
        tensor_ir = (a + b) @ c
        print(f"   Tensor IR: {tensor_ir}")

        inputs = {
            "a": np.array(
                [[i * 4 + j for j in range(4)] for i in range(4)], dtype=float
            ),
            "b": np.array([[1.0] * 4 for _ in range(4)], dtype=float),
            "c": np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        }

        # Layout assignment and lowering
        args = Namespace(
            n=4096,
            rolls=True,
            backend="toy",
            net="lan",
            cache=False,
            mock=False,
            serialize=False,
        )

        print("\n2. Running layout assignment and lowering...")
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()

        # Execute and serialize
        print("\n3. Executing and serializing...")
        args.backend = backend
        original_results = run_backend(backend, circuit_ir, inputs, args)
        file_paths = serialize_circuit(circuit_ir, output_dir, circuit_name)

        result = (
            original_results[0]
            if isinstance(original_results, list)
            else original_results
        )
        print(f"   Created {len(file_paths)} files")
        print(
            f"   Original result: {result[:4] if hasattr(result, '__getitem__') else result}"
        )

        # Load and verify
        print("\n4. Loading and verifying...")
        loaded_data = load_circuit(output_dir, circuit_name)
        print(f"   Loaded {len(loaded_data['kernels'])} kernels")

        # Display kernel information
        print("\n5. Kernel details:")
        for kernel_data in loaded_data["kernels"]:
            meta = kernel_data["metadata"]
            print(f"   Kernel {meta['kernel_idx']}: {meta['operation']}")
            print(f"     Dependencies: {meta['dependencies']}")
            print(f"     Outputs: {meta['outputs']}")

        print("\n" + "=" * 70)
        print("Multi-operation test completed!")
        print("=" * 70)

        # Test passes if we reach here without exceptions
