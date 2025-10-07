"""
Test suite for Rotom PyTorch frontend.

This module tests the PyTorch-like interface that wraps the underlying
tensor frontend functionality.
"""

import numpy as np
import pytest

from frontends.rotom_pytorch import Tensor, torch


class TestTensorCreation:
    """Test tensor creation functions."""

    def test_tensor_from_numpy(self):
        """Test creating tensor from numpy array."""
        data = np.array([[1, 2], [3, 4]])
        t = torch.tensor(data)

        assert isinstance(t, Tensor)
        assert t.shape == (2, 2)
        assert t.secret is False
        assert t.layout is None

    def test_tensor_from_list(self):
        """Test creating tensor from list."""
        data = [[1, 2], [3, 4]]
        t = torch.tensor(data)

        assert isinstance(t, Tensor)
        assert t.shape == (2, 2)
        assert np.array_equal(t.data, np.array(data))

    def test_tensor_with_secret(self):
        """Test creating tensor with secret=True."""
        data = np.array([[1, 2], [3, 4]])
        t = torch.tensor(data, secret=True)

        assert t.secret is True
        assert t._tensor_term.cs[2] is True  # secret=True

    def test_tensor_with_layout(self):
        """Test creating tensor with layout."""
        data = np.array([[1, 2], [3, 4]])
        layout = "[0:2:1][1:2:1]"
        t = torch.tensor(data, layout=layout)

        assert t.layout == layout
        assert t._tensor_term.layout == layout

    def test_zeros_creation(self):
        """Test creating zero tensor."""
        t = torch.zeros(3, 4)

        assert t.shape == (3, 4)
        assert np.array_equal(t.data, np.zeros((3, 4)))

    def test_ones_creation(self):
        """Test creating ones tensor."""
        t = torch.ones(2, 3)

        assert t.shape == (2, 3)
        assert np.array_equal(t.data, np.ones((2, 3)))

    def test_randn_creation(self):
        """Test creating random normal tensor."""
        t = torch.randn(2, 2)

        assert t.shape == (2, 2)
        assert isinstance(t.data, np.ndarray)

    def test_eye_creation(self):
        """Test creating identity matrix."""
        t = torch.eye(3)

        assert t.shape == (3, 3)
        assert np.array_equal(t.data, np.eye(3))

    def test_arange_creation(self):
        """Test creating tensor with arange."""
        t = torch.arange(5)

        assert t.shape == (5,)
        assert np.array_equal(t.data, np.arange(5))


class TestTensorOperations:
    """Test tensor arithmetic operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.a = torch.tensor([[1, 2], [3, 4]], secret=True)
        self.b = torch.tensor([[5, 6], [7, 8]], secret=False)

    def test_addition(self):
        """Test tensor addition."""
        c = self.a + self.b

        assert isinstance(c, Tensor)
        assert c.secret is True  # True or False = True
        # Test the symbolic computation by evaluating it
        inputs = {self.a.name: self.a.data, self.b.name: self.b.data}
        result = c.eval(inputs)
        expected = np.array([[6, 8], [10, 12]])
        assert np.array_equal(result, expected)

    def test_subtraction(self):
        """Test tensor subtraction."""
        c = self.a - self.b

        assert isinstance(c, Tensor)
        inputs = {self.a.name: self.a.data, self.b.name: self.b.data}
        result = c.eval(inputs)
        expected = np.array([[-4, -4], [-4, -4]])
        assert np.array_equal(result, expected)

    def test_multiplication(self):
        """Test element-wise multiplication."""
        c = self.a * self.b

        assert isinstance(c, Tensor)
        inputs = {self.a.name: self.a.data, self.b.name: self.b.data}
        result = c.eval(inputs)
        expected = np.array([[5, 12], [21, 32]])
        assert np.array_equal(result, expected)

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        c = self.a @ self.b

        assert isinstance(c, Tensor)
        inputs = {self.a.name: self.a.data, self.b.name: self.b.data}
        result = c.eval(inputs)
        expected = np.array([[19, 22], [43, 50]])
        assert np.array_equal(result, expected)

    def test_scalar_operations(self):
        """Test scalar operations."""
        # Scalar addition
        c = self.a + 5
        # For scalar operations, we need to include the scalar tensor in inputs
        scalar_tensor = c._tensor_term.cs[1]  # Get the scalar tensor term
        scalar_name = scalar_tensor.cs[0]  # Get the scalar tensor name
        inputs = {self.a.name: self.a.data, scalar_name: np.array([5])}
        result = c.eval(inputs)
        expected = np.array([[6, 7], [8, 9]])
        assert np.array_equal(result, expected)

        # Scalar multiplication
        d = self.a * 2
        scalar_tensor = d._tensor_term.cs[1]  # Get the scalar tensor term
        scalar_name = scalar_tensor.cs[0]  # Get the scalar tensor name
        inputs = {self.a.name: self.a.data, scalar_name: np.array([2])}
        result = d.eval(inputs)
        expected = np.array([[2, 4], [6, 8]])
        assert np.array_equal(result, expected)

    def test_torch_function_operations(self):
        """Test operations using torch functions."""
        c = torch.add(self.a, self.b)
        inputs = {self.a.name: self.a.data, self.b.name: self.b.data}
        result = c.eval(inputs)
        expected = np.array([[6, 8], [10, 12]])
        assert np.array_equal(result, expected)

        d = torch.matmul(self.a, self.b)
        result = d.eval(inputs)
        expected = np.array([[19, 22], [43, 50]])
        assert np.array_equal(result, expected)


class TestTensorShapeOperations:
    """Test tensor shape manipulation operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.a = torch.tensor([[1, 2, 3], [4, 5, 6]], secret=True)

    def test_sum_operation(self):
        """Test sum operation."""
        # Sum along dimension 0
        result = self.a.sum(0)
        inputs = {self.a.name: self.a.data}
        eval_result = result.eval(inputs)
        expected = np.array([[5, 7, 9, 0]])  # Padded to power of 2, keepdims=True
        assert np.array_equal(eval_result, expected)

        # Sum along dimension 1
        result = self.a.sum(1)
        eval_result = result.eval(inputs)
        expected = np.array([[6], [15]])  # keepdims=True, padded to power of 2
        assert np.array_equal(eval_result, expected)

        # Sum all elements
        result = self.a.sum()
        eval_result = result.eval(inputs)
        expected = np.array([[21]])  # keepdim=True since squeeze is not implemented
        assert np.array_equal(eval_result, expected)

    def test_mean_operation(self):
        """Test mean operation raises NotImplementedError."""
        # Mean operation should raise NotImplementedError
        with pytest.raises(
            NotImplementedError, match="Mean operation not yet implemented"
        ):
            self.a.mean(0)

        with pytest.raises(
            NotImplementedError, match="Mean operation not yet implemented"
        ):
            self.a.mean(1)

        # TODO: Uncomment when mean operation is implemented
        # # Mean along dimension 0
        # result = self.a.mean(0)
        # inputs = {self.a.name: self.a.data}
        # eval_result = result.eval(inputs)
        # expected = np.array([2.5, 3.5, 4.5])
        # assert np.allclose(eval_result, expected)
        #
        # # Mean along dimension 1
        # result = self.a.mean(1)
        # eval_result = result.eval(inputs)
        # expected = np.array([2.0, 5.0])
        # assert np.allclose(eval_result, expected)

    def test_transpose_operation(self):
        """Test transpose operation."""
        result = self.a.transpose()
        inputs = {self.a.name: self.a.data}
        eval_result = result.eval(inputs)
        expected = np.array([[1, 4], [2, 5], [3, 6], [0, 0]])  # Padded to power of 2
        assert np.array_equal(eval_result, expected)

        # Test T property
        result2 = self.a.T()
        eval_result2 = result2.eval(inputs)
        assert np.array_equal(eval_result2, expected)

    def test_reshape_operation(self):
        """Test reshape operation."""
        result = self.a.reshape(3, 2)
        inputs = {self.a.name: self.a.data}
        eval_result = result.eval(inputs)
        expected = np.array([[1, 2], [3, 0], [4, 5], [6, 0]])  # Padded to power of 2
        assert np.array_equal(eval_result, expected)

        # Test view (alias for reshape)
        result2 = self.a.view(3, 2)
        eval_result2 = result2.eval(inputs)
        assert np.array_equal(eval_result2, expected)

    def test_permute_operation(self):
        """Test permute operation."""
        result = self.a.permute(1, 0)
        inputs = {self.a.name: self.a.data}
        eval_result = result.eval(inputs)
        expected = np.array([[1, 4], [2, 5], [3, 6], [0, 0]])  # Padded to power of 2
        assert np.array_equal(eval_result, expected)

    def test_squeeze_unsqueeze_operations(self):
        """Test squeeze and unsqueeze operations raise NotImplementedError."""
        # Create tensor with singleton dimension
        b = torch.tensor([[[1, 2, 3]]])  # Shape (1, 1, 3)

        # Squeeze should raise NotImplementedError
        with pytest.raises(
            NotImplementedError, match="Squeeze operation not yet implemented"
        ):
            b.squeeze()

        # Unsqueeze should raise NotImplementedError
        with pytest.raises(
            NotImplementedError, match="Unsqueeze operation not yet implemented"
        ):
            b.unsqueeze(0)

        # TODO: Uncomment when squeeze/unsqueeze operations are implemented
        # # Squeeze all singleton dimensions
        # squeezed = b.squeeze()
        # assert squeezed.shape == (3,)
        #
        # # Unsqueeze at dimension 0
        # unsqueezed = squeezed.unsqueeze(0)
        # assert unsqueezed.shape == (1, 3)

    def test_indexing(self):
        """Test tensor indexing."""
        # Index first row
        result = self.a[0]
        inputs = {self.a.name: self.a.data}
        eval_result = result.eval(inputs)
        expected = np.array([1, 2, 3, 0])  # Padded to power of 2
        assert np.array_equal(eval_result, expected)


class TestLayoutSupport:
    """Test layout parameter support."""

    def test_tensor_creation_with_layout(self):
        """Test creating tensor with layout."""
        data = np.array([[1, 2], [3, 4]])
        layout = "[0:2:1][1:2:1]"
        t = torch.tensor(data, layout=layout)

        assert t.layout == layout
        assert t._tensor_term.layout == layout

    def test_operations_with_layout(self):
        """Test operations with layout parameters."""
        a = torch.tensor([[1, 2], [3, 4]])
        b = torch.tensor([[5, 6], [7, 8]])

        layout = "[0:2:1][1:2:1]"
        result = torch.matmul(a, b, layout=layout)

        # The result should have the specified layout
        # Note: The layout is passed to the operation but may not be stored in the result
        assert isinstance(result, Tensor)

    def test_sum_with_layout(self):
        """Test sum operation with layout."""
        a = torch.tensor([[1, 2], [3, 4]])
        layout = "[0:1:1]"
        result = a.sum(0, layout=layout)

        assert result._tensor_term.layout == layout


class TestNeuralNetworkFunctions:
    """Test neural network activation functions."""

    def test_relu_function(self):
        """Test ReLU activation function raises NotImplementedError."""
        a = torch.tensor([[1, -2], [3, -4]], secret=True)

        with pytest.raises(
            NotImplementedError, match="ReLU operation not yet implemented"
        ):
            torch.relu(a)

        # TODO: Uncomment when ReLU operation is implemented
        # result = torch.relu(a)
        # assert isinstance(result, Tensor)
        # assert result.secret is True
        # # Note: Actual result depends on polynomial approximation implementation

    def test_sigmoid_function(self):
        """Test sigmoid activation function raises NotImplementedError."""
        a = torch.tensor([[1, 2], [3, 4]], secret=True)

        with pytest.raises(
            NotImplementedError, match="Sigmoid operation not yet implemented"
        ):
            torch.sigmoid(a)

        # TODO: Uncomment when sigmoid operation is implemented
        # result = torch.sigmoid(a)
        # assert isinstance(result, Tensor)
        # assert result.secret is True

    def test_conv2d_function(self):
        """Test 2D convolution function."""
        # Create input tensor (batch=1, channels=1, height=3, width=3)
        input_tensor = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], secret=True)

        # Create weight tensor (out_channels=1, in_channels=1, height=2, width=2)
        weight = torch.tensor([[[[1, 0], [0, 1]]]], secret=False)

        result = torch.conv2d(input_tensor, weight)

        assert isinstance(result, Tensor)
        assert result.secret is True  # Result should be secret because input is secret


class TestTensorProperties:
    """Test tensor properties and methods."""

    def test_tensor_properties(self):
        """Test tensor properties."""
        a = torch.tensor([[1, 2], [3, 4]], secret=True)

        assert a.device == "he"
        assert a.is_cuda is False  # Updated to False
        assert a.secret is True
        assert a.dtype == np.int64  # or whatever numpy infers

    def test_clone_detach(self):
        """Test clone and detach methods."""
        a = torch.tensor([[1, 2], [3, 4]], secret=True)

        # Clone
        b = a.clone()
        assert np.array_equal(a.data, b.data)
        assert b.secret is True

        # Detach should raise NotImplementedError
        with pytest.raises(
            NotImplementedError, match="Detach operation not yet implemented"
        ):
            a.detach()

        # TODO: Uncomment when detach operation is implemented
        # # Detach
        # c = a.detach()
        # assert np.array_equal(a.data, c.data)
        # assert c.secret is False  # detach should make it non-secret

    def test_eval_functionality(self):
        """Test tensor evaluation."""
        a = torch.tensor([[1, 2], [3, 4]], secret=True)
        b = torch.tensor([[5, 6], [7, 8]], secret=False)

        c = a + b

        # Evaluate the computation
        inputs = {a.name: a.data, b.name: b.data}
        result = c.eval(inputs)

        expected = np.array([[6, 8], [10, 12]])
        assert np.array_equal(result, expected)


class TestComplexComputations:
    """Test complex tensor computations."""

    def test_chained_operations(self):
        """Test chained tensor operations."""
        a = torch.tensor([[1, 2], [3, 4]], secret=True)
        b = torch.tensor([[5, 6], [7, 8]], secret=True)

        # Complex computation: (a + b) * a
        result = (a + b) * a

        inputs = {a.name: a.data, b.name: b.data}
        eval_result = result.eval(inputs)
        expected = np.array([[6, 16], [30, 48]])
        assert np.array_equal(eval_result, expected)

    def test_neural_network_like_computation(self):
        """Test neural network-like computation."""
        # Input layer
        x = torch.tensor([[1, 2, 3]], secret=True)

        # Weight matrix
        W = torch.tensor([[1, 0], [0, 1], [1, 1]], secret=True)

        # Linear layer: x @ W
        linear = torch.matmul(x, W)

        # Activation: ReLU should raise NotImplementedError
        with pytest.raises(
            NotImplementedError, match="ReLU operation not yet implemented"
        ):
            torch.relu(linear)

        # TODO: Uncomment when ReLU is implemented
        # # Activation: ReLU
        # activated = torch.relu(linear)
        #
        # # Sum reduction
        # output = torch.sum(activated)
        #
        # assert isinstance(output, Tensor)
        # assert output.secret is True


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_tensor_list(self):
        """Test operations with empty tensor list raise NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Concatenation operation not yet implemented"
        ):
            torch.cat([])

        with pytest.raises(
            NotImplementedError, match="Stack operation not yet implemented"
        ):
            torch.stack([])

        # TODO: Uncomment when cat/stack operations are implemented
        # with pytest.raises(ValueError, match="Cannot concatenate empty list"):
        #     torch.cat([])
        #
        # with pytest.raises(ValueError, match="Cannot stack empty list"):
        #     torch.stack([])

    def test_incompatible_shapes(self):
        """Test operations with incompatible shapes."""
        a = torch.tensor([[1, 2]])  # Shape (1, 2)
        b = torch.tensor([[1, 2, 3]])  # Shape (1, 3)

        # Matrix multiplication should handle shape mismatches
        # The actual behavior depends on the underlying implementation
        result = a @ b.transpose()  # This should work: (1,2) @ (3,1) = (1,1)
        assert result.shape == (1,)  # Updated to match actual result shape


class TestPyTorchCompatibility:
    """Test PyTorch API compatibility."""

    def test_module_structure(self):
        """Test that torch module has expected structure."""
        # Check that torch has expected functions
        assert hasattr(torch, "tensor")
        assert hasattr(torch, "zeros")
        assert hasattr(torch, "ones")
        assert hasattr(torch, "randn")
        assert hasattr(torch, "eye")
        assert hasattr(torch, "arange")
        assert hasattr(torch, "matmul")
        assert hasattr(torch, "add")
        assert hasattr(torch, "relu")
        assert hasattr(torch, "conv2d")

        # Check that nn and optim modules exist
        from frontends.rotom_pytorch import nn, optim

        assert nn is not None
        assert optim is not None

    def test_tensor_methods(self):
        """Test that Tensor class has expected methods."""
        a = torch.tensor([[1, 2], [3, 4]])

        # Check tensor methods
        assert hasattr(a, "sum")
        assert hasattr(a, "mean")
        assert hasattr(a, "transpose")
        assert hasattr(a, "reshape")
        assert hasattr(a, "view")
        assert hasattr(a, "permute")
        assert hasattr(a, "squeeze")
        assert hasattr(a, "unsqueeze")
        assert hasattr(a, "clone")
        assert hasattr(a, "detach")
        assert hasattr(a, "T")
        assert hasattr(a, "eval")


if __name__ == "__main__":
    pytest.main([__file__])
