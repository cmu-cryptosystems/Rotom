Tensor Operations
=================

Rotom supports a wide range of tensor operations optimized for homomorphic encryption. This guide covers the available operations and their usage patterns.

Element-wise Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from frontends.tensor import TensorTerm
   
   # Create tensors
   a = TensorTerm.Tensor("a", [64, 64], True)  # Ciphertext
   b = TensorTerm.Tensor("b", [64, 64], False) # Plaintext
   
   # Addition
   c = a + b
   
   # Subtraction  
   d = a - b
   
   # Multiplication (element-wise)
   e = a * b

Matrix Operations
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Matrix multiplication
   c = a @ b
   
   # Transpose
   d = a.T

Reduction Operations
~~~~~~~~~~~~~~~~~~~~

Reduction operations aggregate values along specified dimensions:

.. code-block:: python

   # Sum along dimension 0
   c = a.sum(0)
   
   # Sum along dimension 1  
   d = a.sum(1)

Shape Operations
~~~~~~~~~~~~~~~~

Shape manipulation operations allow you to restructure tensors:

.. code-block:: python

   # Reshape tensor
   c = a.reshape(0, {0: 32, 1: 128})
   
   # Permute dimensions
   d = a.permute({0: 1, 1: 0})
   
   # Index tensor
   e = a[i]  # where i is an int 


Convolution
~~~~~~~~~~~

Convolution is still a work in progress. We currently support a basic implementation. Future work will extend this implementation to support different strides and paddings. 

.. code-block:: python

   # 2D Convolution
   input_tensor = TensorTerm.Tensor("input", [32, 32, 3], True)
   filter_tensor = TensorTerm.Tensor("filter", [3, 3, 3, 64], False)
   
   output = TensorTerm.conv2d(input_tensor, filter_tensor, 
                             stride=1, padding="same")

