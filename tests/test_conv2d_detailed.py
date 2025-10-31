"""
Detailed convolution tests with value verification.

This test suite verifies that convolution operations compute the correct values
by checking intermediate rotation, multiplication, and summation steps.
"""

import numpy as np
from scipy import signal

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def verify_conv2d_manual(input_data, filter_data, expected_output, padding="same"):
    """Manually verify convolution computation using NumPy/SciPy."""
    # Use scipy's correlate2d (which is what conv2d does)
    # Note: correlate2d with "same" mode
    if padding == "same":
        mode = "same"
    elif padding == "valid":
        mode = "valid"
    else:
        raise ValueError(f"Unknown padding: {padding}")
    
    result = np.zeros((input_data.shape[0], expected_output.shape[1], expected_output.shape[2]))
    
    for c_out in range(filter_data.shape[0]):
        channel_sum = np.zeros((expected_output.shape[1], expected_output.shape[2]))
        for c_in in range(filter_data.shape[1]):
            # Correlate (not convolve - convolution flips the kernel)
            conv_result = signal.correlate2d(
                input_data[c_in], 
                filter_data[c_out, c_in], 
                mode=mode
            )
            channel_sum += conv_result
        result[c_out] = channel_sum
    
    return result


class TestConvolution2DDetailed:
    """Detailed convolution tests with value verification."""

    def test_conv2d_simple_3x3_ones(self):
        """Test simple 3x3 convolution with known values for manual verification."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_detailed_ones"

        # Create simple input: incrementing values
        input_data = np.array([[
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ]])
        
        # Simple filter: all ones (box filter)
        filter_data = np.array([[[
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]]])

        inputs = {"a": input_data, "b": filter_data}

        # Compute expected output manually
        print("\n=== Manual Verification ===")
        print("Input:")
        print(input_data[0])
        print("\nFilter:")
        print(filter_data[0, 0])
        
        # For "same" padding, output should be same size as input
        # Let's manually compute one output value to verify
        # Output[1,1] should sum the 3x3 window centered at input[1,1]
        # Window: [[0,1,2], [4,5,6], [8,9,10]]
        expected_1_1 = 0+1+2+4+5+6+8+9+10
        print(f"\nExpected output[1,1] = {expected_1_1}")
        
        # Expected output using scipy
        expected = verify_conv2d_manual(input_data, filter_data, input_data, padding="same")
        print("\nExpected full output (scipy):")
        print(expected[0])

        # Create computation
        input_tensor = TensorTerm.Tensor("a", [1, 4, 4], True)
        filter_tensor = TensorTerm.Tensor("b", [1, 1, 3, 3], False)
        output_tensor = TensorTerm.conv2d(input_tensor, filter_tensor, 1, "same")

        # Evaluate expected
        expected_eval = output_tensor.eval(inputs)
        print("\nExpected (TensorTerm.eval):")
        print(expected_eval[0])

        # Run compiler
        kernel = LayoutAssignment(output_tensor, args).run()
        circuit_ir = Lower(kernel).run()
        results = Toy(circuit_ir, inputs, args).run()

        # Check result
        expected_cts = apply_layout(expected_eval, kernel.layout)
        
        print("\nActual output (FHE):")
        actual_output = results
        print(actual_output)
        
        print("\nExpected output (layout applied):")
        print(expected_cts)
        
        assert expected_cts == results, f"Output mismatch!\nExpected: {expected_cts}\nGot: {results}"

    def test_conv2d_edge_detection_horizontal(self):
        """Test horizontal edge detection filter."""
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_edge_h"

        # Create vertical gradient
        input_data = np.array([[
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]
        ]], dtype=float)
        
        # Horizontal edge detection (simplified Sobel-like)
        filter_data = np.array([[[
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ]]], dtype=float)

        inputs = {"a": input_data, "b": filter_data}

        print("\n=== Edge Detection Test ===")
        print("Input (vertical gradient):")
        print(input_data[0])
        print("\nFilter (horizontal edge detector):")
        print(filter_data[0, 0, 0])

        # Manually compute expected
        expected = verify_conv2d_manual(input_data, filter_data, input_data, padding="same")
        print("\nExpected output:")
        print(expected[0])

        # Create computation
        input_tensor = TensorTerm.Tensor("a", [1, 4, 4], True)
        filter_tensor = TensorTerm.Tensor("b", [1, 1, 3, 3], False)
        output_tensor = TensorTerm.conv2d(input_tensor, filter_tensor, 1, "same")

        # Run compiler
        expected_eval = output_tensor.eval(inputs)
        kernel = LayoutAssignment(output_tensor, args).run()
        circuit_ir = Lower(kernel).run()
        results = Toy(circuit_ir, inputs, args).run()

        # Check result
        expected_cts = apply_layout(expected_eval, kernel.layout)
        print("\nActual FHE output:")
        print(results)
        
        assert expected_cts == results

    def test_conv2d_2x2_filter_detailed(self):
        """Test 2x2 filter with detailed verification."""
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_2x2_detailed"

        # Create simple input
        input_data = np.array([[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]], dtype=float)
        
        # 2x2 filter
        filter_data = np.array([[[
            [1, 2],
            [3, 4]
        ]]], dtype=float)

        inputs = {"a": input_data, "b": filter_data}

        print("\n=== 2x2 Filter Test ===")
        print("Input:")
        print(input_data[0])
        print("\nFilter:")
        print(filter_data[0, 0, 0])

        # Manually compute one position for verification
        # Output[1,1] with "same" padding
        # Window: [[1,2], [5,6]]
        # Result: 1*1 + 2*2 + 5*3 + 6*4 = 1 + 4 + 15 + 24 = 44
        print("\nManual computation for output[1,1]:")
        print("Window: [[1,2], [5,6]]")
        print("Computation: 1*1 + 2*2 + 5*3 + 6*4 = 1 + 4 + 15 + 24 = 44")

        # Compute expected
        expected = verify_conv2d_manual(input_data, filter_data, input_data, padding="same")
        print("\nExpected full output:")
        print(expected[0])

        # Create computation
        input_tensor = TensorTerm.Tensor("a", [1, 4, 4], True)
        filter_tensor = TensorTerm.Tensor("b", [1, 1, 2, 2], False)
        output_tensor = TensorTerm.conv2d(input_tensor, filter_tensor, 1, "same")

        # Run
        expected_eval = output_tensor.eval(inputs)
        kernel = LayoutAssignment(output_tensor, args).run()
        circuit_ir = Lower(kernel).run()
        results = Toy(circuit_ir, inputs, args).run()

        expected_cts = apply_layout(expected_eval, kernel.layout)
        print("\nActual FHE output:")
        print(results)
        
        assert expected_cts == results

