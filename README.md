# Rotom

Rotom is a tensor compiler that autovectorizes high-level programs into homomorphic encryption circuits—searching layouts, applying roll-style transforms, and reducing rotation cost.

The compiler can target **[HEIR](https://heir.dev/)**, Google's homomorphic encryption stack in MLIR (*Homomorphic Encryption Intermediate Representation*). With `--backend heir`, Rotom writes HEIR-compatible MLIR plus serialized inputs; a bundled MLIR interpreter can check correctness locally before you run HEIR's compiler passes.

- **Website & layout visualizer:** [cmu-cryptosystems.github.io/Rotom](https://cmu-cryptosystems.github.io/Rotom/)

Maintained by [CMU Cryptosystems](https://github.com/cmu-cryptosystems).

### Installation

```bash
# Clone the repository
git clone https://github.com/cmu-cryptosystems/Rotom.git
cd Rotom 

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import argparse
import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from lower.lower import Lower

# Create tensor operations
a = TensorTerm.Tensor("a", [64, 64], True)  # 64x64 ciphertext matrix
b = TensorTerm.Tensor("b", [64], False)  # 64-element plaintext vector
c = a @ b  # Matrix-vector multiplication

# Run layout assignment, lower to circuit IR, execute with toy backend
args = argparse.Namespace(n=4096, rolls=True, backend="toy")
kernel = LayoutAssignment(c, args).run()
circuit_ir = Lower(kernel).run()
inputs = {"a": np.random.randn(64, 64), "b": np.random.randn(64)}
results = Toy(circuit_ir, inputs, args).run()
```

### Google HEIR backend

To lower compiled circuits to **HEIR** MLIR instead of running the toy or OpenFHE backends:

```bash
python main.py --backend heir --n 4096 --rolls --fn <name>
```

Output is written under `heir/<name>/` (MLIR, packed inputs, and optional results). You can feed that MLIR into HEIR’s compiler passes; see [backends/heir/README.md](backends/heir/README.md) for layout, the bundled interpreter, and upstream integration notes ([HEIR repository](https://github.com/google/heir), [example PR](https://github.com/google/heir/pull/2432)).

### Running Tests

```bash
# Unit tests
pytest

# Specific benchmarks
python main.py --benchmark matmul_128_64 --backend toy --rolls
```

### Downloading Benchmarks

Rotom includes wrappers for running benchmarks from FHELIPE and Viaduct compilers. To use these benchmarks, you need to download them from the [rotom_benchmarks repository](https://github.com/edwjchen/rotom_benchmarks):

```bash
# Clone the benchmarks repository into the Rotom directory
cd Rotom
git clone https://github.com/edwjchen/rotom_benchmarks.git benchmarks

# Or if you prefer a different location, you can clone it elsewhere
# and reference it with absolute paths when running benchmarks
```

The benchmarks should be placed in `Rotom/benchmarks/` with the following structure:
- `benchmarks/fhelipe_benchmarks/` - FHELIPE benchmarks
- `benchmarks/viaduct_benchmarks/` - Viaduct benchmarks

For more details about the available benchmarks, see the [benchmarks README](https://github.com/edwjchen/rotom_benchmarks).

### Running FHELIPE and Viaduct Benchmarks

Rotom includes wrappers for running benchmarks from FHELIPE and Viaduct compilers. These wrappers parse the compiler output and execute the circuits using Rotom's CKKS backend.

#### FHELIPE Benchmarks

```bash
# Run a FHELIPE benchmark
python main.py --fhelipe <path_to_benchmark_directory> --n <slots>

# Example: Run double matrix multiplication benchmark with 32768 slots
python main.py --fhelipe benchmarks/fhelipe_benchmarks/double_matmul_128_64 --n 32768

# Use --not-secure flag to allow smaller ring dimensions (for testing)
python main.py --fhelipe benchmarks/fhelipe_benchmarks/double_matmul_128_64 --n 4096 --not-secure
```

**Note**: For 128-bit security, `n` must be at least 16384. Use `--not-secure` for testing with smaller values.

#### Viaduct Benchmarks

```bash
# Run a Viaduct benchmark
python main.py --viaduct <path_to_benchmark_file> --n <slots>

# Example: Run double matrix multiplication benchmark with 32768 slots
python main.py --viaduct benchmarks/viaduct_benchmarks/double_matmul_ct_pt_e1_o0.vhe --n 32768

# Use --not-secure flag to allow smaller ring dimensions (for testing)
python main.py --viaduct benchmarks/viaduct_benchmarks/distance_e1_o0.vhe --n 4096 --not-secure
```

## Architecture Overview

```
High-level Tensor Operations
           ↓
    Frontend (tensor.py)
           ↓
    IR (layout.py, dim.py, roll.py)
           ↓
    Layout Assignment (assignment.py)
           ↓
    Lowering (lower.py)
           ↓
    Backend: toy / OpenFHE (CKKS) / HEIR MLIR (heir.py)
           ↓
    Simulated HE, runtime FHE, or MLIR for Google HEIR
```

## Core Components

### Frontend (`frontends/`)
- Tensor operation definitions
- Shape inference and validation
- High-level API for users

### IR (`ir/`)
- Dimension definitions (`dim.py`)
- Roll operations (`roll.py`)
- Layout representation (`layout.py`)
- Kernel operations (`kernel.py`)
- Cost modeling (`kernel_cost.py`)

### Layout Assignment (`assignment/`)

### Backends (`backends/`)
- **Toy** — plaintext simulation of the circuit IR
- **OpenFHE** — CKKS execution (`openfhe_backend.py`)
- **HEIR** — MLIR emission and tooling for [Google HEIR](https://github.com/google/heir) (`backends/heir/`)

### Utilities (`util/`)
- Layout utilities (`layout_util.py`)
- Kernel utilities (`kernel_util.py`)
- Shape utilities (`shape_util.py`)


## Development

### Adding New Tensor Operations

1. Add operation to `frontends/tensor.py`
2. Create generator in `assignment/gen/`
3. Add lowering pass in `lower/`
4. Add backend support if needed
5. Create tests in `tests/`

### Adding New Backends

1. Create backend class in `backends/`
2. Implement required methods
3. Add command line argument
4. Update `main.py`
5. Add tests and benchmarks

## Documentation

This project includes comprehensive documentation generated using Sphinx. The documentation covers user guides, API references, and detailed explanations of Rotom's concepts.

### Documentation Structure

- **User Guide**: Comprehensive guides for using Rotom
  - Writing Tensor Programs
  - Understanding Layout Representations
  - Tensor Operations
  - Backend Configuration
- **API Reference**: Complete API documentation for all modules
- **Architecture Overview**: Detailed explanation of Rotom's compilation process

### Generating Documentation

```bash
# Navigate to docs directory
cd docs

# Generate HTML documentation
python gen_docs.py

# Clean and regenerate
python gen_docs.py --clean
```

### Viewing Documentation

```bash
cd docs
python serve_docs.py 8000
# Then open http://localhost:8000 in your browser
```

### Documentation Files

- `docs/source/` - Sphinx source files and configuration
- `docs/html/` - Generated HTML documentation
- `docs/gen_docs.py` - Documentation generation script
- `docs/serve_docs.py` - Local web server for viewing docs
- `docs/source/writing_tensor_programs.rst` - Guide for writing tensor programs
- `docs/source/understanding_layout_representations.rst` - Layout system explanation
- `docs/source/user_guide/` - User guide documentation
- `docs/source/api_reference/` - API reference documentation


## Citation

If you use Rotom in research, please cite:

**Edward Chen**, **Fraser Brown**, **Wenting Zheng**. *Bridging Usability and Performance: A Tensor Compiler for Autovectorizing Homomorphic Encryption.* [IACR ePrint 2025/1319](https://eprint.iacr.org/2025/1319). Published at USENIX Security 2026.

```bibtex
@misc{cryptoeprint:2025/1319,
    author = {Edward Chen and Fraser Brown and Wenting Zheng},
    title = {Bridging Usability and Performance: A Tensor Compiler for Autovectorizing Homomorphic Encryption},
    howpublished = {Cryptology {ePrint} Archive, Paper 2025/1319},
    year = {2025},
    url = {https://eprint.iacr.org/2025/1319}
}
```

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

