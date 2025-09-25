Writing Tensor Programs
==================================

This guide shows you how to write your own tensor programs using Rotom's high-level tensor interface.

Tensor Terms
---------------------

Tensor terms represent the basic building blocks. You can create public tensors (plaintext data) and secret tensors (encrypted data) using a ``TensorTerm.Tensor``:

.. code-block:: python

   from frontends.tensor import TensorTerm

   # Create secret tensor (encrypted data)
   a = TensorTerm.Tensor("a", [64, 64], secret=True)  # 64x64 ciphertext matrix
   
   # Create public tensor (unencrypted data)
   b = TensorTerm.Tensor("b", [64], secret=False)     # 64-element plaintext vector


Complete Example
--------------------------------------------

.. code-block:: python

   import argparse
   import numpy as np

   from frontends.tensor import TensorTerm
   from assignment.assignment import LayoutAssignment
   from lower.lower import Lower
   from backends.toy import Toy
   from util.layout_util import apply_layout

   # Create tensor terms
   a = TensorTerm.Tensor("a", [64, 64], secret=True)  # 64x64 ciphertext matrix
   b = TensorTerm.Tensor("b", [64], secret=False)     # 64-element plaintext vector

   # Create computation
   c = a @ b  # Matrix-vector multiplication

   # Create test inputs
   inputs = {"a": np.random.randint(0, 2, (64, 64)), "b": np.random.randint(0, 2, (64))}
   args = argparse.Namespace(
      n=4096,
      rolls=True,
      backend="toy",
      net="lan",
   )

   # LayoutAssignment: Optimizes tensor layouts for efficient homomorphic encryption
   # This stage determines how tensors are packed into ciphertext slots and optimizes
   # rotations and other HE operations for the target backend
   assignment = LayoutAssignment(c, args)
   kernel = assignment.run()

   # Lower: Converts optimized tensor operations to low-level HE operations
   # This stage translates high-level tensor operations into a circuit representation
   # that can be executed on HE backends, handling encryption/decryption boundaries
   lower = Lower(kernel)
   circuit = lower.run()

   # Toy: Simulates HE execution using a toy backend for testing and validation
   # This backend provides a reference implementation for testing correctness
   # without the overhead of actual homomorphic encryption
   toy_backend = Toy(circuit, inputs, args)
   results = toy_backend.run()

   # Verify results using check_results function
   expected_cts = apply_layout(c.eval(inputs), kernel.layout)
   assert expected_cts == results

   print("✓ Matrix vector multiplication test passed!")

For more advanced examples and detailed API documentation, see the :doc:`examples` and :doc:`api_reference/index` sections.
