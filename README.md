# Rotom 

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
from frontends.tensor import TensorTerm
from assignment.assignment import LayoutAssignment
import argparse

# Create tensor operations
a = TensorTerm.Tensor("a", [64, 64], True)  # 64x64 ciphertext matrix
b = TensorTerm.Tensor("b", [64], False)     # 64-element plaintext vector
c = a @ b  # Matrix-vector multiplication

# Run layout assignment
args = argparse.Namespace(n=4096, rolls=True, backend="toy")
assignment = LayoutAssignment(c, args)
kernel = assignment.run()

# Execute with toy backend
from backends.toy import Toy
results = Toy(circuit_ir, inputs, args).run()
```

### Running Tests

```bash
# Unit tests
pytest

# Integration tests
python main.py

# Specific benchmarks
python main.py --benchmark matmul --backend toy --rolls
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
    Backend (toy.py, openfhe_backend.py)
           ↓
    HE Circuit
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
- Code generation for HE libraries
- Runtime execution

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


## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

