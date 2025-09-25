Understanding Layout Representations
=============================================

This document provides a comprehensive guide to understanding how Rotom represents and visualizes layouts for tensors. Layout representations are crucial for understanding how tensor data is packed into HE vectors and how various operations like rotations and permutations affect the data layout.

Overview
--------

In homomorphic encryption, tensor data must be carefully packed into HE vectors to enable efficient computation. Rotom uses a sophisticated layout system that tracks:

- **Dimension ordering**: How tensor dimensions are arranged in memory
- **Roll operations**: Permutations that rearrange data within HE vectors  
- **Slot mapping**: How tensor elements map to specific slots in HE vectors
- **Ciphertext distribution**: How large tensors are distributed across multiple ciphertexts

Key Concepts
------------

Dimension
~~~~~~~~~~~~~~~~~~~~~~~~

Dimensions in Rotom are represented using the ``Dim`` class, which captures:

- **Extent**: The size of the dimension (must be a power of 2)
- **Stride**: The step size when traversing the dimension
- **Type**: Whether the dimension is filled with data or empty (zero-filled)

Dimension syntax: ``[dim_index:extent:stride]``

Examples:
- ``[0:4:1]``: Dimension 0, extent 4, stride 1
- ``[1:2:4]``: Dimension 1, extent 2, stride 4  
- ``[4]``: Fixed extent of 4 (dimension index None)


Dimension to Index Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Understanding how tensor indices map to packed vector positions is crucial for layout comprehension.

For a 4x4 matrix with tensor indices (row, col):

**Row-Major Layout Mapping:**
::

   Tensor Index (row, col) → Vector Index
   (0, 0) → 0    (0, 1) → 1    (0, 2) → 2    (0, 3) → 3
   (1, 0) → 4    (1, 1) → 5    (1, 2) → 6    (1, 3) → 7
   (2, 0) → 8    (2, 1) → 9    (2, 2) → 10   (2, 3) → 11
   (3, 0) → 12   (3, 1) → 13   (3, 2) → 14   (3, 3) → 15

**Column-Major Layout Mapping:**
::

   Tensor Index (row, col) → Vector Index
   (0, 0) → 0    (1, 0) → 4    (2, 0) → 8    (3, 0) → 12
   (0, 1) → 1    (1, 1) → 5    (2, 1) → 9    (3, 1) → 13
   (0, 2) → 2    (1, 2) → 6    (2, 2) → 10   (3, 2) → 14
   (0, 3) → 3    (1, 3) → 7    (2, 3) → 11   (3, 3) → 15

**General Formula:**
For layout ``[dim_0:extent_0:stride_0][dim_1:extent_1:stride_1]...``:

- **Row-major**: ``index = row * extent_1 + col``
- **Column-major**: ``index = col * extent_0 + row``

**3D Tensor Example:**
For a 2x4x4 tensor with indices (k, j, i):

**Layout [0:2:1][1:4:1][2:4:1] mapping:**
::

   Tensor Index (k, j, i) → Vector Index
   (0, 0, 0) → 0    (0, 0, 1) → 1    (0, 0, 2) → 2    (0, 0, 3) → 3
   (0, 1, 0) → 4    (0, 1, 1) → 5    (0, 1, 2) → 6    (0, 1, 3) → 7
   ...
   (1, 3, 0) → 28   (1, 3, 1) → 29   (1, 3, 2) → 30   (1, 3, 3) → 31


Layout Representation
~~~~~~~~~~~~~~~~

A ``Layout`` object contains:

- **Dimensions**: List of ``Dim`` objects in their packing order
- **Rolls**: List of ``Roll`` operations applied to the layout
- **Slot count**: Total number of slots in the HE vector (n)
- **Secret flag**: Whether this is ciphertext or plaintext data

Basic Layout Examples
---------------------

Simple Row-Major Layout
~~~~~~~~~~~~~~~~~~~~~~~

For a 4x4 matrix with row-major ordering:

**Original 4x4 matrix:**
::

   [[ 0  1  2  3]
    [ 4  5  6  7]
    [ 8  9 10 11]
    [12 13 14 15]]

**Row-major layout:** ``[0:4:1][1:4:1]``

**Packed vector:**
::

   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

Column-Major Layout
~~~~~~~~~~~~~~~~~~~

For the same matrix with column-major ordering:

**Column-major layout:** ``[1:4:1][0:4:1]``

**Packed vector:**
::

   [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]

Roll Operations
~~~~~~~~~~~~~~~

Roll operations permute data within HE vectors. They are represented as ``Roll(from_dim, to_dim)``:

Diagonal Layout with Roll
~~~~~~~~~~~~~~~~~~~~~~~~~

To create a diagonal layout:

**Layout with roll:** ``roll(0,1) [0:4:1][1:4:1]``

**Packed vector:**
::

   [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]

This creates a diagonal pattern where elements are arranged in a specific permutation that's useful for certain HE operations.

Complex Layout Examples
-----------------------

Multi-Dimensional Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~

For a 2x4x4 tensor:

**Original 3D tensor:**
::

   [[[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]

    [[16 17 18 19]
     [20 21 22 23]
     [24 25 26 27]
     [28 29 30 31]]]

**3D layout:** ``[0:2:1][1:4:1][2:4:1]``

**Packed vector:**
::

   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

Ciphertext Distribution
~~~~~~~~~~~~~~~~~~~~~~~

Large tensors may span multiple ciphertexts. The layout shows how data is distributed:

**Ciphertext distribution layout:** ``[R:4:1];[0:4:1][1:4:1]``

This creates 4 ciphertexts, each containing the same 4x4 matrix data. The ``[R:4:1]`` dimension indicates ciphertext replication.

**Result:** 4 ciphertexts, each containing:
::

   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


Layout Visualization
--------------------

Understanding Layout Strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Layout strings show the dimension structure and any applied rolls:

- ``[0:4:1][1:4:1]``: Simple row-major, no rolls
- ``[1:4:1][0:4:1]``: Simple column-major, no rolls
- ``roll(0,1) [1:4:1][0:4:1]``: Column-major with roll from dim 0 to dim 1

Slot Mapping
~~~~~~~~~~~~

Each position in the HE vector corresponds to a specific tensor element:

.. code-block:: text

   HE Vector Index:    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
   Row-Major Layout:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15]
   Column-Major:      [0, 4, 8,12, 1, 5, 9,13, 2, 6,10,14, 3, 7,11,15]
   Diagonal Roll:     [0, 5,10,15, 1, 6,11,12, 2, 7, 8,13, 3, 4, 9,14]


Layout Visualization Tools
==========================

Rotom provides a layout visualizer tool to help understand how different layouts affect data packing. The visualizer creates test tensors and shows how they are packed into HE vectors.

Using the Layout Visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The layout visualizer can be used to experiment with different layouts:

.. code-block:: python

   from layout_visualizer import visualize_layout, demo_layout_examples
   
   # Visualize a specific layout
   layout, packed = visualize_layout("[0:4:1][1:4:1]", 16, (4, 4))
   
   # Run demonstration examples
   demo_layout_examples()

The visualizer shows:

- Original tensor data
- Layout string representation  
- Packed vector(s) in HE format
- Multiple ciphertexts when using replication

Layout Construction from Strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also create layout objects directly from string representations:

.. code-block:: python

   from ir.layout import Layout
   
   # Create layouts from strings
   row_major = Layout.from_string("[0:4:1][1:4:1]", 16)
   column_major = Layout.from_string("[1:4:1][0:4:1]", 16)
   with_roll = Layout.from_string("roll(0,1) [0:4:1][1:4:1]", 16)
   with_ciphertext = Layout.from_string("[R:4:1];[0:4:1][1:4:1]", 16)

This makes it easy to experiment with different layouts and understand their effects on data packing.
