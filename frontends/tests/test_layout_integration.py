"""
Integration tests for tensor layout functionality.

This module contains integration tests that demonstrate realistic usage
of the layout parameter functionality in tensor computations.
"""

import pytest

from frontends.tensor import TensorOp, TensorTerm


class TestLayoutIntegration:
    """Integration tests for layout functionality."""

    def test_matrix_multiplication_with_different_layouts(self):
        """Test matrix multiplication with different input and output layouts."""
        # Create matrices with different layouts
        a = TensorTerm.Tensor("a", [4, 4], True, layout="[0:4:1][1:4:1]")  # Row-major
        b = TensorTerm.Tensor(
            "b", [4, 4], True, layout="[1:4:1][0:4:1]"
        )  # Column-major

        # Matrix multiplication with specific output layout
        result = a.matmul(b, layout="[0:4:1][1:4:1]")

        assert result.op == TensorOp.MATMUL
        assert result.layout == "[0:4:1][1:4:1]"
        assert result.cs[0] == a
        assert result.cs[1] == b

    def test_neural_network_layer_simulation(self):
        """Test a simulated neural network layer with layout specifications."""
        # Input tensor (batch_size, input_features)
        input_tensor = TensorTerm.Tensor(
            "input", [32, 128], True, layout="[0:32:1][1:128:1]"
        )

        # Weight matrix (input_features, output_features)
        weights = TensorTerm.Tensor(
            "weights", [128, 64], False, layout="[0:128:1][1:64:1]"
        )

        # Bias vector (output_features,)
        bias = TensorTerm.Tensor("bias", [64], False, layout="[0:64:1]")

        # Linear layer: input @ weights + bias
        linear_output = input_tensor.matmul(weights, layout="[0:32:1][1:64:1]")
        layer_output = linear_output + bias

        assert linear_output.layout == "[0:32:1][1:64:1]"
        assert layer_output.layout is None  # No layout specified for addition

    def test_convolutional_layer_with_layout(self):
        """Test convolutional layer with layout specifications."""
        # Input tensor (channels, height, width)
        input_tensor = TensorTerm.Tensor(
            "input", [3, 32, 32], True, layout="[0:3:1][1:32:1][2:32:1]"
        )

        # Filter tensor (output_channels, input_channels, filter_h, filter_w)
        filters = TensorTerm.Tensor(
            "filters", [64, 3, 3, 3], False, layout="[0:64:1][1:3:1][2:3:1][3:3:1]"
        )

        # Convolution with specific output layout
        conv_output = TensorTerm.conv2d(
            input_tensor, filters, 1, "same", layout="[0:64:1][1:32:1][2:32:1]"
        )

        assert conv_output.op == TensorOp.CONV2D
        assert conv_output.layout == "[0:64:1][1:32:1][2:32:1]"

    def test_residual_connection_with_layouts(self):
        """Test residual connection with layout specifications."""
        # Main branch tensor
        main_branch = TensorTerm.Tensor("main", [4, 4], True, layout="[0:4:1][1:4:1]")

        # Residual branch tensor
        residual = TensorTerm.Tensor("residual", [4, 4], True, layout="[0:4:1][1:4:1]")

        # Residual connection: main + residual
        output = main_branch.__add__(residual, layout="[0:4:1][1:4:1]")

        assert output.layout == "[0:4:1][1:4:1]"
        assert output.op == TensorOp.ADD

    def test_attention_mechanism_simulation(self):
        """Test a simplified attention mechanism with layout specifications."""
        # Query, Key, Value tensors (batch_size, seq_len, d_model)
        query = TensorTerm.Tensor(
            "query", [8, 16, 64], True, layout="[0:8:1][1:16:1][2:64:1]"
        )
        key = TensorTerm.Tensor(
            "key", [8, 16, 64], True, layout="[0:8:1][1:16:1][2:64:1]"
        )
        value = TensorTerm.Tensor(
            "value", [8, 16, 64], True, layout="[0:8:1][1:16:1][2:64:1]"
        )

        # Attention scores: query @ key^T
        attention_scores = query.matmul(
            key.transpose(layout="[0:8:1][2:64:1][1:16:1]"),
            layout="[0:8:1][1:16:1][1:16:1]",
        )

        # Apply attention to values
        attended_values = attention_scores.matmul(
            value, layout="[0:8:1][1:16:1][2:64:1]"
        )

        assert attention_scores.layout == "[0:8:1][1:16:1][1:16:1]"
        assert attended_values.layout == "[0:8:1][1:16:1][2:64:1]"

    def test_batch_normalization_simulation(self):
        """Test batch normalization with layout specifications."""
        # Input tensor (batch_size, features)
        input_tensor = TensorTerm.Tensor(
            "input", [32, 128], True, layout="[0:32:1][1:128:1]"
        )

        # Normalize along batch dimension (dimension 0)
        normalized = input_tensor.sum(0, layout="[0:1:1]")  # Sum along batch dim

        # Apply learned scale and shift
        scale = TensorTerm.Tensor("scale", [128], False, layout="[0:128:1]")
        shift = TensorTerm.Tensor("shift", [128], False, layout="[0:128:1]")

        # Scale and shift operations
        scaled = normalized.__mul__(scale, layout="[0:128:1]")
        output = scaled.__add__(shift, layout="[0:128:1]")

        assert normalized.layout == "[0:1:1]"
        assert scaled.layout == "[0:128:1]"
        assert output.layout == "[0:128:1]"

    def test_multi_head_attention_layouts(self):
        """Test multi-head attention with different layouts for each head."""
        # Input tensor (batch_size, seq_len, d_model)
        input_tensor = TensorTerm.Tensor(
            "input", [8, 16, 128], True, layout="[0:8:1][1:16:1][2:128:1]"
        )

        # Split into multiple heads (simplified)
        head1 = input_tensor.reshape(2, {2: 64}, layout="[0:8:1][1:16:1][2:64:1]")
        head2 = input_tensor.reshape(2, {2: 64}, layout="[0:8:1][1:16:1][2:64:1]")

        # Different attention patterns for each head
        attention1 = head1.transpose(layout="[0:8:1][2:64:1][1:16:1]")
        attention2 = head2.transpose(layout="[0:8:1][1:16:1][2:64:1]")

        assert head1.layout == "[0:8:1][1:16:1][2:64:1]"
        assert head2.layout == "[0:8:1][1:16:1][2:64:1]"
        assert attention1.layout == "[0:8:1][2:64:1][1:16:1]"
        assert attention2.layout == "[0:8:1][1:16:1][2:64:1]"


class TestLayoutPerformanceScenarios:
    """Test layout functionality in performance-critical scenarios."""

    def test_large_tensor_operations_with_layouts(self):
        """Test layout functionality with larger tensor dimensions."""
        # Large matrices
        a = TensorTerm.Tensor("a", [256, 256], True, layout="[0:256:1][1:256:1]")
        b = TensorTerm.Tensor("b", [256, 256], True, layout="[0:256:1][1:256:1]")

        # Matrix multiplication with specific layout
        result = a.matmul(b, layout="[0:256:1][1:256:1]")

        assert result.layout == "[0:256:1][1:256:1]"
        assert result.op == TensorOp.MATMUL

    def test_chained_operations_with_mixed_layouts(self):
        """Test chained operations with mixed layout specifications."""
        # Start with a tensor
        a = TensorTerm.Tensor("a", [8, 8], True, layout="[0:8:1][1:8:1]")

        # Chain of operations with different layouts
        step1 = a.transpose(layout="[1:8:1][0:8:1]")  # Transpose with layout
        step2 = step1 + a  # Addition without layout
        step3 = step2.matmul(a, layout="[0:8:1][1:8:1]")  # Matmul with layout
        step4 = step3.sum(0)  # Sum without layout
        step5 = step4.transpose(layout="[0:1:1]")  # Final transpose with layout

        # Verify layout chain
        assert a.layout == "[0:8:1][1:8:1]"
        assert step1.layout == "[1:8:1][0:8:1]"
        assert step2.layout is None
        assert step3.layout == "[0:8:1][1:8:1]"
        assert step4.layout is None
        assert step5.layout == "[0:1:1]"


if __name__ == "__main__":
    pytest.main([__file__])
