"""
Test suite for MLIR interpreter.

This module tests the MLIR interpreter functionality used by the HEIR backend
to verify generated MLIR code.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from backends.heir.mlir_interpreter import (
    interpret_mlir,
    read_input_vector,
    run_mlir_interpreter,
)


class TestMLIRInterpreter:
    """Test MLIR interpreter operations."""

    def setup_method(self):
        """Set up temporary directory for test MLIR files."""
        self.test_dir = tempfile.mkdtemp()
        self.inputs_dir = os.path.join(self.test_dir, "inputs")
        os.makedirs(self.inputs_dir, exist_ok=True)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_input_file(self, filename, values):
        """Write input vector file in numpy compressed format."""
        filepath = os.path.join(self.inputs_dir, filename)
        np.savez_compressed(filepath, data=np.array(values, dtype=np.float32))
        return filepath

    def _write_mlir_file(self, content, filename="test.mlir"):
        """Write MLIR file to test directory."""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def test_read_input_vector(self):
        """Test reading input vector from file."""
        values = [1.0, 2.0, 3.0, 4.0]
        filepath = self._write_input_file("test.npz", values)

        result = read_input_vector(filepath)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, np.array(values, dtype=np.float32))

    def test_read_input_vector_empty(self):
        """Test reading empty input vector."""
        filepath = self._write_input_file("empty.npz", [])

        result = read_input_vector(filepath)

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_interpret_mlir_simple_add(self):
        """Test interpreting simple addition MLIR."""
        # Create input files
        self._write_input_file("2.npz", [1.0, 2.0, 3.0, 4.0])
        self._write_input_file("3.npz", [5.0, 6.0, 7.0, 8.0])

        # Create MLIR file
        mlir_content = """func.func @test(%2 : tensor<4xf32>, %3 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = arith.addf %2, %3 : tensor<4xf32>
  return %1 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = interpret_mlir(mlir_file, self.inputs_dir)

        assert result is not None
        expected = np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_interpret_mlir_simple_multiply(self):
        """Test interpreting simple multiplication MLIR."""
        self._write_input_file("2.npz", [1.0, 2.0, 3.0, 4.0])
        self._write_input_file("3.npz", [2.0, 3.0, 4.0, 5.0])

        mlir_content = """func.func @test(%2 : tensor<4xf32>, %3 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = arith.mulf %2, %3 : tensor<4xf32>
  return %1 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = interpret_mlir(mlir_file, self.inputs_dir)

        assert result is not None
        expected = np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_interpret_mlir_simple_subtract(self):
        """Test interpreting simple subtraction MLIR."""
        self._write_input_file("2.npz", [10.0, 8.0, 6.0, 4.0])
        self._write_input_file("3.npz", [1.0, 2.0, 3.0, 4.0])

        mlir_content = """func.func @test(%2 : tensor<4xf32>, %3 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = arith.subf %2, %3 : tensor<4xf32>
  return %1 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = interpret_mlir(mlir_file, self.inputs_dir)

        assert result is not None
        expected = np.array([9.0, 6.0, 3.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_interpret_mlir_constant(self):
        """Test interpreting MLIR with constants."""
        self._write_input_file("2.npz", [1.0, 2.0, 3.0, 4.0])

        mlir_content = """func.func @test(%2 : tensor<4xf32>) -> tensor<4xf32> {
  %c1 = arith.constant 2 : index
  %c2 = arith.constant dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf32>
  %1 = arith.addf %2, %c2 : tensor<4xf32>
  return %1 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = interpret_mlir(mlir_file, self.inputs_dir)

        assert result is not None
        expected = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_interpret_mlir_rotate(self):
        """Test interpreting MLIR with rotation operation."""
        self._write_input_file("2.npz", [1.0, 2.0, 3.0, 4.0])

        mlir_content = """func.func @test(%2 : tensor<4xf32>) -> tensor<4xf32> {
  %c1 = arith.constant 1 : index
  %1 = tensor_ext.rotate %2, %c1 : tensor<4xf32>, index
  return %1 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = interpret_mlir(mlir_file, self.inputs_dir)

        assert result is not None
        # Rotate left by 1: [1,2,3,4] -> [2,3,4,1]
        expected = np.array([2.0, 3.0, 4.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_interpret_mlir_rotate_negative(self):
        """Test interpreting MLIR with negative rotation."""
        self._write_input_file("2.npz", [1.0, 2.0, 3.0, 4.0])

        mlir_content = """func.func @test(%2 : tensor<4xf32>) -> tensor<4xf32> {
  %c1 = arith.constant 3 : index
  %1 = tensor_ext.rotate %2, %c1 : tensor<4xf32>, index
  return %1 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = interpret_mlir(mlir_file, self.inputs_dir)

        assert result is not None
        # Rotate left by 3: [1,2,3,4] -> [4,1,2,3]
        expected = np.array([4.0, 1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_interpret_mlir_complex_expression(self):
        """Test interpreting complex MLIR expression with multiple operations."""
        self._write_input_file("2.npz", [1.0, 2.0, 3.0, 4.0])
        self._write_input_file("3.npz", [2.0, 3.0, 4.0, 5.0])

        mlir_content = """func.func @test(%2 : tensor<4xf32>, %3 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = arith.mulf %2, %3 : tensor<4xf32>
  %4 = arith.addf %1, %2 : tensor<4xf32>
  %5 = arith.subf %4, %3 : tensor<4xf32>
  return %5 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = interpret_mlir(mlir_file, self.inputs_dir)

        assert result is not None
        # %1 = %2 * %3 = [2, 6, 12, 20]
        # %4 = %1 + %2 = [3, 8, 15, 24]
        # %5 = %4 - %3 = [1, 5, 11, 19]
        expected = np.array([1.0, 5.0, 11.0, 19.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_run_mlir_interpreter(self):
        """Test run_mlir_interpreter wrapper function."""
        self._write_input_file("2.npz", [1.0, 2.0, 3.0, 4.0])
        self._write_input_file("3.npz", [5.0, 6.0, 7.0, 8.0])

        mlir_content = """func.func @test(%2 : tensor<4xf32>, %3 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = arith.addf %2, %3 : tensor<4xf32>
  return %1 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = run_mlir_interpreter(mlir_file, self.inputs_dir)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        expected = [6.0, 8.0, 10.0, 12.0]
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_run_mlir_interpreter_file_not_found(self):
        """Test run_mlir_interpreter with non-existent file."""
        with pytest.raises(RuntimeError, match="MLIR interpretation failed"):
            run_mlir_interpreter("nonexistent.mlir")

    def test_interpret_mlir_file_not_found(self):
        """Test interpret_mlir with non-existent file."""
        with pytest.raises(FileNotFoundError, match="MLIR file not found"):
            interpret_mlir("nonexistent.mlir")

    def test_interpret_mlir_no_return(self):
        """Test interpret_mlir with MLIR that has no return statement."""
        self._write_input_file("2.npz", [1.0, 2.0, 3.0, 4.0])

        mlir_content = """func.func @test(%2 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = arith.addf %2, %2 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = interpret_mlir(mlir_file, self.inputs_dir)

        # Should return None when no return statement is found
        assert result is None

    def test_interpret_mlir_default_inputs_dir(self):
        """Test that interpret_mlir uses default inputs/ directory."""
        self._write_input_file("2.npz", [1.0, 2.0, 3.0, 4.0])

        mlir_content = """func.func @test(%2 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = arith.addf %2, %2 : tensor<4xf32>
  return %1 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = interpret_mlir(mlir_file)

        assert result is not None
        expected = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_interpret_mlir_multiple_inputs(self):
        """Test interpreting MLIR with multiple input parameters."""
        self._write_input_file("2.npz", [1.0, 2.0])
        self._write_input_file("5.npz", [3.0, 4.0])
        self._write_input_file("7.npz", [5.0, 6.0])

        mlir_content = """func.func @test(%2 : tensor<2xf32>, %5 : tensor<2xf32>, %7 : tensor<2xf32>) -> tensor<2xf32> {
  %1 = arith.addf %2, %5 : tensor<2xf32>
  %3 = arith.addf %1, %7 : tensor<2xf32>
  return %3 : tensor<2xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        result = interpret_mlir(mlir_file, self.inputs_dir)

        assert result is not None
        # %1 = %2 + %5 = [4, 6]
        # %3 = %1 + %7 = [9, 12]
        expected = np.array([9.0, 12.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_run_mlir_interpreter_returns_none(self):
        """Test run_mlir_interpreter when interpret_mlir returns None."""
        self._write_input_file("2.npz", [1.0, 2.0, 3.0, 4.0])

        # MLIR with no return statement
        mlir_content = """func.func @test(%2 : tensor<4xf32>) -> tensor<4xf32> {
  %1 = arith.addf %2, %2 : tensor<4xf32>
}
"""
        mlir_file = self._write_mlir_file(mlir_content)

        # The exception is wrapped in RuntimeError
        with pytest.raises(
            RuntimeError,
            match="MLIR interpretation failed.*MLIR interpretation returned None",
        ):
            run_mlir_interpreter(mlir_file, self.inputs_dir)
