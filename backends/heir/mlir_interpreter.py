"""
MLIR Interpreter for executing HEIR-generated MLIR code.

This module provides functionality to parse and execute MLIR code generated
by the HEIR backend, allowing verification of the generated MLIR against
expected results.
"""

import os
import re

import numpy as np


def read_input_vector(fn):
    """
    Read a vector from input file (first line is length, then values).
    
    Args:
        fn: Path to the input file
        
    Returns:
        numpy.ndarray: The vector as a numpy array
    """
    with open(fn, 'r') as f:
        length = int(f.readline().strip())
        values = []
        for _ in range(length):
            values.append(float(f.readline().strip()))
        return np.array(values, dtype=np.float32)


def interpret_mlir(mlir_file, inputs_dir=None):
    """
    Interpret the generated MLIR file and execute it.
    
    Args:
        mlir_file: Path to the MLIR file to interpret
        inputs_dir: Directory containing input files (defaults to inputs/ relative to MLIR file)
        
    Returns:
        numpy.ndarray: The computed result vector, or None if no result found
    """
    if not os.path.exists(mlir_file):
        raise FileNotFoundError(f"MLIR file not found: {mlir_file}")
    
    # Determine inputs directory
    if inputs_dir is None:
        # Default to inputs/ directory relative to MLIR file
        mlir_dir = os.path.dirname(mlir_file)
        inputs_dir = os.path.join(mlir_dir, "inputs")
    
    # Read and parse MLIR
    with open(mlir_file, 'r') as f:
        lines = f.readlines()
    
    # Parse function signature to get inputs
    func_line = lines[0].strip()
    # Extract parameter names and types
    # Format: func.func @test(%2 : tensor<16xf32> {secret.secret}, %3 : tensor<16xf32>, ...) -> tensor<16xf32>
    
    # Read input vectors
    input_vectors = {}
    
    # Parse function signature to map parameters correctly
    # Extract parameters: %2 : tensor<16xf32> {secret.secret}, %3 : tensor<16xf32>, ...
    param_match = re.search(r'\(([^)]+)\)', func_line)
    if param_match:
        params_str = param_match.group(1)
        params = [p.strip() for p in params_str.split(',')]
        for param in params:
            # Extract variable name (e.g., %2 from "%2 : tensor<16xf32>")
            var_match = re.match(r'(%\d+)', param)
            if var_match:
                var_name = var_match.group(1)
                # Try to read input file by SSA variable name first (e.g., %2 -> inputs/2.txt)
                var_filename = var_name.lstrip('%')
                input_file = os.path.join(inputs_dir, f"{var_filename}.txt")
                if os.path.exists(input_file):
                    input_vectors[var_name] = read_input_vector(input_file)
    
    # Execution environment
    env = {}
    env.update(input_vectors)
    
    step = 1
    # Parse and execute operations
    for line in lines[1:]:  # Skip function signature
        line = line.strip()
        if not line or line.startswith('}') or line.startswith('//'):
            continue
        
        # Parse operation
        # Format: %result = op %arg1, %arg2 : type
        # or: %result = op %arg1, %arg2 : type1, type2
        
        # Match: %var = operation args : types
        match = re.match(r'(%\w+) = (\w+\.\w+) (.+) : (.+)', line)
        if not match:
            # Try return statement
            if line.startswith('return'):
                return_match = re.search(r'return (%\w+)', line)
                if return_match:
                    result_var = return_match.group(1)
                    result = env.get(result_var, None)
                    if result is not None:
                        return result
            continue
        
        result_var = match.group(1)
        operation = match.group(2)
        args_str = match.group(3).strip()
        types_str = match.group(4).strip()
        
        # Parse arguments
        args = [arg.strip() for arg in args_str.split(',')]

        # Execute operation
        if operation == 'arith.constant':
            # Format: %c5 = arith.constant 4 : index
            const_match = re.match(r'(\d+)', args_str)
            if const_match:
                value = int(const_match.group(1))
                env[result_var] = value
            else:
                # Dense constant
                dense_match = re.search(r'dense<([^>]+)>', args_str)
                if dense_match:
                    # Parse dense array
                    values_str = dense_match.group(1)
                    values = [float(x.strip()) for x in values_str.split(',')]
                    env[result_var] = np.array(values, dtype=np.float32)
        
        elif operation == 'arith.mulf':
            # Multiplication: %1 = arith.mulf %2, %3 : tensor<16xf32>
            arg1 = env.get(args[0])
            arg2 = env.get(args[1])
            if arg1 is not None and arg2 is not None:
                result = arg1 * arg2
                env[result_var] = result
        
        elif operation == 'arith.addf':
            # Addition: %8 = arith.addf %1, %6 : tensor<16xf32>
            arg1 = env.get(args[0])
            arg2 = env.get(args[1])
            if arg1 is not None and arg2 is not None:
                result = arg1 + arg2
                env[result_var] = result
        
        elif operation == 'arith.subf':
            # Subtraction
            arg1 = env.get(args[0])
            arg2 = env.get(args[1])
            if arg1 is not None and arg2 is not None:
                result = arg1 - arg2
                env[result_var] = result
        
        elif operation == 'tensor_ext.rotate':
            # Rotation: %4 = tensor_ext.rotate %2, %c5 : tensor<16xf32>, index
            # positive rotation amount means left rotate
            arg1 = env.get(args[0])
            rot_amt = env.get(args[1])
            if arg1 is not None and rot_amt is not None:
                vector = arg1.copy()
                n = len(vector)
                # Left rotate by rot_amt: element at i goes to (i - rot_amt) % n
                rotated = np.array([vector[(i + rot_amt) % n] for i in range(n)])
                env[result_var] = rotated
        
        step += 1
    
    # Get return value - look for return statement
    for line in lines:
        line = line.strip()
        if line.startswith('return'):
            # Match: return %16 : tensor<16xf32>
            return_match = re.search(r'return (%\w+)', line)
            if return_match:
                result_var = return_match.group(1)
                result = env.get(result_var, None)
                if result is not None:
                    return result
            # Also try without % prefix
            return_match = re.search(r'return (\w+)', line)
            if return_match:
                result_var = return_match.group(1)
                result = env.get(result_var, None)
                if result is not None:
                    return result
    
    return None


def run_mlir_interpreter(mlir_file, inputs_dir=None):
    """
    Run the MLIR interpreter on the generated MLIR file.
    Returns the computed results as a list of vectors, compatible with check_results.
    
    Args:
        mlir_file: Path to the MLIR file to interpret
        inputs_dir: Directory containing input files (defaults to inputs/ relative to MLIR file)
    
    Returns:
        list: List of result vectors from MLIR execution
        
    Raises:
        RuntimeError: If MLIR interpretation fails
        ValueError: If MLIR interpretation returns None
    """
    try:
        mlir_result = interpret_mlir(mlir_file, inputs_dir)
        
        if mlir_result is None:
            raise ValueError("MLIR interpretation returned None")
        
        # Convert to list of vectors format expected by check_results
        # MLIR typically returns a single result vector
        if mlir_result.ndim == 1:
            return [mlir_result.tolist()]
        else:
            return [mlir_result.tolist()]
    
    except Exception as e:
        raise RuntimeError(f"MLIR interpretation failed: {str(e)}") from e

