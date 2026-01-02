"""
Test suite for LoLa MNIST model layers.

This module tests each layer of the LoLa MNIST architecture to verify
that packing works correctly for each operation.

Model Architecture:
- Input: 28×28×1
- Conv1: 5×5 kernel, 5 filters, stride 2 → 14×14×5
- Activation: x² (square)
- Conv2: 5×5 kernel, 50 filters, stride 2 → 7×7×50
- Activation: x² (square)
- Flatten: 2450 neurons
- FC1: 2450 → 100
- Activation: x² (square)
- FC2: 100 → 10
- Output: logits (no final activation)
"""

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestLoLaMNISTLayers:
    """Test individual layers of the LoLa MNIST architecture."""

    def _run_test_case(self, tensor_ir, inputs, args):
        """Helper method to run a test case."""
        # Generate expected result
        expected = tensor_ir.eval(inputs)

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = Toy(circuit_ir, inputs, args).run()

        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert expected_cts == results

    def test_conv1_layer(self):
        """Test Conv1 layer: 28×28×1 → 14×14×5 with 5×5 kernel, stride 2."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "lola_conv1"

        # Create inputs
        # Input: 1 channel, 28×28 (C, H, W format)
        input_channels = 1
        input_size = 28
        output_channels = 5
        kernel_size = 5
        stride = 2
        padding = "same"  # Assuming "same" padding for stride 2

        inputs = {}
        # Create random binary input for FHE compatibility
        np.random.seed(42)
        inputs["a"] = np.random.randint(0, 2, (input_channels, input_size, input_size)).astype(float)
        # Weight tensor: [f_out, f_in, f_h, f_w]
        inputs["b"] = np.random.randint(-1, 2, (output_channels, input_channels, kernel_size, kernel_size)).astype(float)

        # Create computation
        input_tensor = TensorTerm.Tensor("a", [input_channels, input_size, input_size], True)
        weight_tensor = TensorTerm.Tensor("b", [output_channels, input_channels, kernel_size, kernel_size], False)
        output_tensor = TensorTerm.conv2d(input_tensor, weight_tensor, stride, padding)

        # Run test
        self._run_test_case(output_tensor, inputs, args)

    def test_conv1_with_square_activation(self):
        """Test Conv1 layer followed by square activation (x²)."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "lola_conv1_square"

        # Create inputs
        input_channels = 1
        input_size = 28
        output_channels = 5
        kernel_size = 5
        stride = 2
        padding = "same"

        inputs = {}
        np.random.seed(42)
        inputs["a"] = np.random.randint(0, 2, (input_channels, input_size, input_size)).astype(float)
        inputs["b"] = np.random.randint(-1, 2, (output_channels, input_channels, kernel_size, kernel_size)).astype(float)

        # Create computation: Conv1 + Square activation
        input_tensor = TensorTerm.Tensor("a", [input_channels, input_size, input_size], True)
        weight_tensor = TensorTerm.Tensor("b", [output_channels, input_channels, kernel_size, kernel_size], False)
        conv_output = TensorTerm.conv2d(input_tensor, weight_tensor, stride, padding)
        # Square activation: x² = x * x
        square_output = conv_output * conv_output

        # Run test
        self._run_test_case(square_output, inputs, args)

    def test_conv2_layer(self):
        """Test Conv2 layer: 14×14×5 → 7×7×50 with 5×5 kernel, stride 2."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "lola_conv2"

        # Create inputs
        input_channels = 5
        input_size = 14
        output_channels = 50
        kernel_size = 5
        stride = 2
        padding = "same"

        inputs = {}
        np.random.seed(43)
        inputs["a"] = np.random.randint(0, 2, (input_channels, input_size, input_size)).astype(float)
        inputs["b"] = np.random.randint(-1, 2, (output_channels, input_channels, kernel_size, kernel_size)).astype(float)

        # Create computation
        input_tensor = TensorTerm.Tensor("a", [input_channels, input_size, input_size], True)
        weight_tensor = TensorTerm.Tensor("b", [output_channels, input_channels, kernel_size, kernel_size], False)
        output_tensor = TensorTerm.conv2d(input_tensor, weight_tensor, stride, padding)

        # Run test
        self._run_test_case(output_tensor, inputs, args)

    def test_conv2_with_square_activation(self):
        """Test Conv2 layer followed by square activation (x²)."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "lola_conv2_square"

        # Create inputs
        input_channels = 5
        input_size = 14
        output_channels = 50
        kernel_size = 5
        stride = 2
        padding = "same"

        inputs = {}
        np.random.seed(43)
        inputs["a"] = np.random.randint(0, 2, (input_channels, input_size, input_size)).astype(float)
        inputs["b"] = np.random.randint(-1, 2, (output_channels, input_channels, kernel_size, kernel_size)).astype(float)

        # Create computation: Conv2 + Square activation
        input_tensor = TensorTerm.Tensor("a", [input_channels, input_size, input_size], True)
        weight_tensor = TensorTerm.Tensor("b", [output_channels, input_channels, kernel_size, kernel_size], False)
        conv_output = TensorTerm.conv2d(input_tensor, weight_tensor, stride, padding)
        # Square activation: x² = x * x
        square_output = conv_output * conv_output

        # Run test
        self._run_test_case(square_output, inputs, args)

    def test_flatten_layer(self):
        """Test Flatten layer: 7×7×50 → 2450.
        
        Flattening requires two reshape steps:
        1. Combine spatial dimensions: [50, 7, 7] → [50, 49]
        2. Combine all dimensions: [50, 49] → [2450]
        """
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "lola_flatten"

        # Create inputs
        channels = 50
        height = 7
        width = 7
        flattened_size = channels * height * width  # 2450

        inputs = {}
        np.random.seed(44)
        inputs["a"] = np.random.randint(0, 2, (channels, height, width)).astype(float)

        # Create computation: Flatten by reshaping in two steps
        input_tensor = TensorTerm.Tensor("a", [channels, height, width], True)
        # Step 1: Combine spatial dimensions [50, 7, 7] → [50, 49]
        height_width = height * width  # 49
        intermediate = input_tensor.reshape(2, {1: height_width})
        # Step 2: Combine all dimensions [50, 49] → [2450]
        flattened_tensor = intermediate.reshape(1, {0: flattened_size})

        # Run test
        self._run_test_case(flattened_tensor, inputs, args)

    def test_fc1_layer(self):
        """Test FC1 layer: 2450 → 100."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "lola_fc1"

        # Create inputs
        input_features = 2450
        output_features = 100

        inputs = {}
        np.random.seed(45)
        # Input is a vector of size 2450
        inputs["a"] = np.random.randint(0, 2, (input_features,)).astype(float)
        # Weight matrix: [input_features, output_features]
        inputs["b"] = np.random.randint(-1, 2, (input_features, output_features)).astype(float)

        # Create computation: FC1 = input @ weights
        # For matmul: if input is 1D (2450,), treat as row vector (1, 2450)
        # Weight is (2450, 100), result will be (1, 100)
        input_tensor = TensorTerm.Tensor("a", [input_features], True)
        weight_tensor = TensorTerm.Tensor("b", [input_features, output_features], False)
        output_tensor = input_tensor @ weight_tensor

        # Run test
        self._run_test_case(output_tensor, inputs, args)

    def test_fc1_with_square_activation(self):
        """Test FC1 layer followed by square activation (x²)."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "lola_fc1_square"

        # Create inputs
        input_features = 2450
        output_features = 100

        inputs = {}
        np.random.seed(45)
        inputs["a"] = np.random.randint(0, 2, (input_features,)).astype(float)
        inputs["b"] = np.random.randint(-1, 2, (input_features, output_features)).astype(float)

        # Create computation: FC1 + Square activation
        input_tensor = TensorTerm.Tensor("a", [input_features], True)
        weight_tensor = TensorTerm.Tensor("b", [input_features, output_features], False)
        fc_output = input_tensor @ weight_tensor
        # Square activation: x² = x * x
        square_output = fc_output * fc_output

        # Run test
        self._run_test_case(square_output, inputs, args)

    def test_fc2_layer(self):
        """Test FC2 layer: 100 → 10."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "lola_fc2"

        # Create inputs
        input_features = 100
        output_features = 10

        inputs = {}
        np.random.seed(46)
        # Input is a vector of size 100
        inputs["a"] = np.random.randint(0, 2, (input_features,)).astype(float)
        # Weight matrix: [input_features, output_features]
        inputs["b"] = np.random.randint(-1, 2, (input_features, output_features)).astype(float)

        # Create computation: FC2 = input @ weights
        input_tensor = TensorTerm.Tensor("a", [input_features], True)
        weight_tensor = TensorTerm.Tensor("b", [input_features, output_features], False)
        output_tensor = input_tensor @ weight_tensor

        # Run test
        self._run_test_case(output_tensor, inputs, args)

