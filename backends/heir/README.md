# HEIR Backend

The HEIR backend generates MLIR code compatible with the [HEIR](https://github.com/google/heir) homomorphic encryption research framework. This backend enables you to compile FHE circuits to HEIR's MLIR format for further processing and execution.

## Overview

The HEIR backend performs the following operations:

1. **Circuit Lowering**: Converts the Rotom circuit IR into HEIR-compatible MLIR
2. **Input Serialization**: Writes input data in a format compatible with HEIR
3. **MLIR Generation**: Produces MLIR code with operations like `arith.mulf`, `arith.addf`, `tensor_ext.rotate`, etc.
4. **Result Verification**: Uses an integrated MLIR interpreter to verify the generated MLIR produces correct results

## Quick Start

### Serialize to MLIR for the HEIR Backend

```bash
python main.py --backend heir --n <number_of_slots> --rolls --fn <name>
```

### Command-Line Arguments

Key arguments for the HEIR backend:

- `--backend heir`: Selects the HEIR backend
- `--n <size>`: Sets the number of slots (vector size) for homomorphic encryption. Common values: 4096, 8192, 16384
- `--rolls`: Enables roll optimizations 
- `--fn <name>`: Sets the output filename and directory name (default: "main")

## Output Structure

The HEIR backend generates output in the `heir/<fn>/` directory with the following structure:

```
heir/
└── <fn>/
    ├── <fn>.mlir              # Generated MLIR code
    ├── base/                   # Base input tensors (by layout term)
    │   └── {input}_{layout}.npz
    ├── inputs/                 # MLIR function inputs (by SSA variable ID)
    │   ├── {SSA_ID}.npz
    │   └── ...
    └── results/                # Execution results
        └── result.npz
```

## MLIR Interpreter

The HEIR backend includes an MLIR interpreter (`backends/heir/mlir_interpreter.py`) that:

- Parses the generated MLIR file
- Reads input vectors from the `inputs/` directory
- Executes MLIR operations
- Returns computed results for verification

## Example: Chained Matrix Multiplication

The running example demonstrates chained matrix multiplication:

```bash
python main.py --backend heir --n 4096 --rolls --fn chained_matmul
```

This computes `a @ b @ c` where:
- `a`, `b`, `c` are 64×64 matrices
- The computation is packed into 4096-element vectors
- Roll optimizations are applied for efficient rotation operations

The output includes:
- MLIR code in `heir/chained_matmul/chained_matmul.mlir`
- Input vectors in `heir/chained_matmul/inputs/`
- Results in `heir/chained_matmul/results/result_0.txt`

## Integration with HEIR Framework

The generated MLIR can be used with the HEIR framework:

1. **Copy the necessary files to HEIR**: Copy the following from `heir/<fn>/` to your HEIR project:
   - `<fn>.mlir` - The generated MLIR file
   - `inputs/` - Directory containing input vectors
   - `results/` - Directory containing expected results (for verification)
2. **Use HEIR's compilation pipeline**: Pass the MLIR through HEIR's passes
3. See example: [https://github.com/google/heir/pull/2432](https://github.com/google/heir/pull/2432)

## See Also

- [HEIR Project](https://github.com/google/heir): The HEIR homomorphic encryption research framework
- Main Rotom documentation in the project root

