Backends
========

Rotom supports multiple backends for different Homomorphic Encryption (HE) libraries. This guide explains the available backends and how to use them effectively.

Overview
--------

Backends are responsible for lowering the HE operations IR to specific HE libraries. Each backend targets a different HE implementation.

Supported Backends
------------------

Toy Backend
~~~~~~~~~~~

The Toy backend provides plaintext simulation for development and testing:

**Use Cases:**

- Development and debugging
- Algorithm verification
- Performance testing without HE overhead
- Educational purposes

**Features:**

- Fast execution (no encryption overhead)
- Full compatibility with HE operations
- Easy debugging and inspection
- No security guarantees


OpenFHE Backend
~~~~~~~~~~~~~~~

The OpenFHE backend generates code for the OpenFHE library, providing production-ready CKKS implementation:

**Use Cases:**

- Real-world applications
- High-performance FHE computations
- CKKS-based applications

**Configuration:**

You can manually configure various cryptographic parameters in the OpenFHE backend.


HEIR Backend
~~~~~~~~~~~~

The lowering pass to HEIR is still a work in progress. 
For more information about HEIR, visit the `HEIR project website <https://heir.dev/>`_.
