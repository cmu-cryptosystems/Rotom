"""
MLIR Interpreter for executing generated MLIR code for HEIR.

This module provides functionality to parse and execute MLIR code generated
by the HEIR backend, allowing verification of the generated MLIR against
expected results.
"""

import os
import re
import struct

import numpy as np


def read_input_vector(fn):
    """
    Read a vector or tensor from numpy compressed file.

    Args:
        fn: Path to the input file (.npz)

    Returns:
        numpy.ndarray: The vector/tensor as a numpy array
    """
    data = np.load(fn)
    return data["data"]


def from_mlir_hex(hex_str: str) -> np.ndarray:
    """Deserialize an MLIR dense hex string back to a numpy array (f32)."""
    hex_data = hex_str[2:]  # strip 0x
    n = len(hex_data) // 8  # 4 bytes per f32 = 8 hex chars
    vals = struct.unpack("<" + "f" * n, bytes.fromhex(hex_data))
    return np.array(vals, dtype=np.float32)


def interpret_mlir(mlir_file, n, inputs_dir=None):
    if not os.path.exists(mlir_file):
        raise FileNotFoundError(f"MLIR file not found: {mlir_file}")

    if inputs_dir is None:
        mlir_dir = os.path.dirname(mlir_file)
        inputs_dir = os.path.join(mlir_dir, "inputs")

    with open(mlir_file, "r") as f:
        content = f.read()
        lines = content.splitlines()

    # Find the main func.func (not private helpers like relu)
    main_func_line = None
    main_func_line_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"func\.func @\w+\(", stripped) and "private" not in stripped:
            main_func_line = stripped
            main_func_line_idx = i
            break

    if main_func_line is None:
        raise ValueError("No main func.func found in MLIR file")

    # Find the end of the main function (matching closing brace)
    # Collect only lines belonging to the main function
    main_func_lines = []
    depth = 0
    in_main_func = False
    for i, line in enumerate(lines):
        if i == main_func_line_idx:
            in_main_func = True
        if in_main_func:
            main_func_lines.append(line.strip())
            depth += line.count("{") - line.count("}")
            if depth == 0 and i > main_func_line_idx:
                break

    # Read input vectors
    input_vectors = {}

    # Parse function signature
    param_match = re.search(r"func\.func @\w+\((.+?)\)\s*->", main_func_line)
    if param_match:
        params_str = param_match.group(1)
        params = re.split(r",\s*(?=%)", params_str)
        for param in params:
            param = param.strip()
            var_match = re.match(r"(%\w+)", param)
            if var_match:
                var_name = var_match.group(1)
                var_filename = var_name.lstrip("%")
                input_file = os.path.join(inputs_dir, f"{var_filename}.npz")
                if os.path.exists(input_file):
                    input_vectors[var_name] = read_input_vector(input_file)

    env = {}
    env.update(input_vectors)

    # Execute operations in the main function body (skip signature and closing brace)
    for line in main_func_lines[1:]:
        line = line.strip()
        if not line or line.startswith("}") or line.startswith("//"):
            continue

        match = re.match(r"(%\w+) = (\w+\.\w+) (.+) : (.+)", line)
        if not match:
            match = re.match(r"(%\w+) = (call) (.+) : (.+)", line)

        if match is None:
            continue

        result_var = match.group(1)
        operation = match.group(2)
        args_str = match.group(3).strip()
        args = [arg.strip() for arg in args_str.split(",")]

        if operation == "arith.constant":
            dense_match = re.search(r"dense<([^>]+)>", args_str)
            if dense_match:
                # constants
                inner = dense_match.group(1).strip()
                if inner.startswith('"'):
                    hex_str = inner.strip('"')
                    env[result_var] = from_mlir_hex(hex_str)
                else:
                    values_str = inner.strip("[]")
                    values = [
                        float(x.strip()) for x in values_str.split(",") if x.strip()
                    ]
                    if len(values) == 1:
                        env[result_var] = [values[0]] * n
                    else:
                        env[result_var] = np.array(values, dtype=np.float32)
            else:
                # rot offset
                const_match = re.match(r"(-?\d+)", args_str)
                if const_match:
                    env[result_var] = int(const_match.group(1))

        elif operation == "arith.mulf":
            arg1 = env.get(args[0])
            arg2 = env.get(args[1])
            if arg1 is None or arg2 is None:
                raise ValueError("arg1 or arg2 is None")
            arg1 = np.asarray(arg1).flatten()
            arg2 = np.asarray(arg2).flatten()
            env[result_var] = np.array([a * b for a, b in zip(arg1, arg2)])

        elif operation == "arith.addf":
            arg1 = env.get(args[0])
            arg2 = env.get(args[1])
            if arg1 is None or arg2 is None:
                raise ValueError("arg1 or arg2 is None")
            env[result_var] = np.array([a + b for a, b in zip(arg1, arg2)])

        elif operation == "arith.subf":
            arg1 = env.get(args[0])
            arg2 = env.get(args[1])
            if arg1 is None or arg2 is None:
                raise ValueError("arg1 or arg2 is None")
            arg1 = np.asarray(arg1).flatten()
            arg2 = np.asarray(arg2).flatten()
            env[result_var] = np.array([a - b for a, b in zip(arg1, arg2)])

        elif operation == "tensor_ext.rotate":
            arg1 = env.get(args[0])
            rot_amt = env.get(args[1])
            if arg1 is None or rot_amt is None:
                raise ValueError("arg1 or rot_amt is None")
            vector = np.asarray(arg1).flatten()
            vec_len = len(vector)
            rotated = np.array(
                [vector[(i + rot_amt) % vec_len] for i in range(vec_len)],
                dtype=np.float32,
            )
            env[result_var] = rotated

        elif operation == "tensor.extract_slice":
            source_match = re.match(r"(%\w+)", args_str)
            if not source_match:
                continue
            source_arg = source_match.group(1)
            source_tensor = env.get(source_arg)

            if source_tensor is not None:
                offsets_match = re.search(r"\[([^\]]+)\]", args_str)
                if offsets_match:
                    offsets_str = offsets_match.group(1)
                    offsets = [int(x.strip()) for x in offsets_str.split(",")]

                    sizes_match = re.search(
                        r"\[([^\]]+)\]", args_str[args_str.find("]") + 1 :]
                    )
                    if sizes_match:
                        sizes_str = sizes_match.group(1)
                        sizes = [
                            int(x.strip())
                            for x in re.split(r"[,\s]+", sizes_str)
                            if x.strip()
                        ]

                        if source_tensor.ndim == 2:
                            row_start, col_start = offsets[0], offsets[1]
                            row_size, col_size = sizes[0], sizes[1]
                            slice_result = source_tensor[
                                row_start : row_start + row_size,
                                col_start : col_start + col_size,
                            ]
                            env[result_var] = (
                                slice_result.flatten()
                                if row_size == 1
                                else slice_result
                            )
                        elif source_tensor.ndim == 1:
                            start = offsets[0]
                            size = sizes[0]
                            env[result_var] = source_tensor[start : start + size]
                        else:
                            raise ValueError(
                                f"Unsupported tensor dimension for extract_slice: {source_tensor.ndim}"
                            )

        elif operation == "call":
            call_match = re.match(r"@(\w+)\((%\w+)\)", args_str)
            if call_match:
                func_name = call_match.group(1)
                arg_var = call_match.group(2)
                arg = env.get(arg_var)
                if arg is not None:
                    if func_name == "relu":
                        env[result_var] = np.maximum(arg, 0)
                    else:
                        raise ValueError(f"Unknown called function: @{func_name}")

    # Get return value
    for line in reversed(main_func_lines):
        line = line.strip()
        if line.startswith("return"):
            return_match = re.search(r"return (%\w+)", line)
            if return_match:
                result_var = return_match.group(1)
                result = env.get(result_var)
                if result is not None:
                    return result

    return None


def run_mlir_interpreter(mlir_file, n, inputs_dir=None):
    """
    Run the MLIR interpreter on the generated MLIR file.
    Returns the computed results as a list of vectors, compatible with check_results.

    Args:
        mlir_file: Path to the MLIR file to interpret.
        n: Vector length used for broadcasting scalar constants.
        inputs_dir: Directory containing input files. If not provided, it
            defaults to ``inputs/`` relative to ``mlir_file``.

    Returns:
        list: List of result vectors from MLIR execution

    Raises:
        RuntimeError: If MLIR interpretation fails.
        ValueError: If MLIR interpretation returns None.
    """
    if inputs_dir is None:
        mlir_dir = os.path.dirname(mlir_file)
        inputs_dir = os.path.join(mlir_dir, "inputs")

    try:
        mlir_result = interpret_mlir(mlir_file, n=n, inputs_dir=inputs_dir)
        if mlir_result is None:
            raise ValueError("MLIR interpretation returned None")
        return [mlir_result.tolist()]

    except Exception as e:
        raise RuntimeError(f"MLIR interpretation failed: {str(e)}") from e
