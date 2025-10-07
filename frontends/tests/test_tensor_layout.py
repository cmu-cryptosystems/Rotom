"""
Test suite for tensor layout functionality.

This module tests the new optional layout parameter functionality
that was added to all tensor operators and methods.
"""

import pytest
from frontends.tensor import TensorTerm, TensorOp


class TestTensorLayoutCreation:
    """Test tensor creation with layout parameters."""
    
    def test_tensor_creation_without_layout(self):
        """Test that tensor creation without layout still works."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        assert a.layout is None
        assert a.op == TensorOp.TENSOR
        assert a.cs == ["a", [4, 4], True]
    
    def test_tensor_creation_with_layout(self):
        """Test tensor creation with layout parameter."""
        layout = "[0:4:1][1:4:1]"
        a = TensorTerm.Tensor("a", [4, 4], True, layout=layout)
        assert a.layout == layout
        assert a.op == TensorOp.TENSOR
        assert a.cs == ["a", [4, 4], True]
    
    def test_const_creation_without_layout(self):
        """Test constant creation without layout."""
        c = TensorTerm.const(42)
        assert c.layout is None
        assert c.op == TensorOp.CONST
        assert c.cs == [42]
    
    def test_const_creation_with_layout(self):
        """Test constant creation with layout."""
        layout = "[0:1:1]"
        c = TensorTerm.const(42, layout=layout)
        assert c.layout == layout
        assert c.op == TensorOp.CONST
        assert c.cs == [42]


class TestTensorOperatorsLayout:
    """Test tensor operators with layout parameters."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.a = TensorTerm.Tensor("a", [4, 4], True)
        self.b = TensorTerm.Tensor("b", [4, 4], True)
        self.layout = "[0:4:1][1:4:1]"
    
    def test_addition_without_layout(self):
        """Test addition without layout (backward compatibility)."""
        result = self.a + self.b
        assert result.layout is None
        assert result.op == TensorOp.ADD
    
    def test_addition_with_layout(self):
        """Test addition with layout parameter."""
        result = self.a.__add__(self.b, layout=self.layout)
        assert result.layout == self.layout
        assert result.op == TensorOp.ADD
    
    def test_subtraction_without_layout(self):
        """Test subtraction without layout."""
        result = self.a - self.b
        assert result.layout is None
        assert result.op == TensorOp.SUB
    
    def test_subtraction_with_layout(self):
        """Test subtraction with layout parameter."""
        result = self.a.__sub__(self.b, layout=self.layout)
        assert result.layout == self.layout
        assert result.op == TensorOp.SUB
    
    def test_multiplication_without_layout(self):
        """Test multiplication without layout."""
        result = self.a * self.b
        assert result.layout is None
        assert result.op == TensorOp.MUL
    
    def test_multiplication_with_layout(self):
        """Test multiplication with layout parameter."""
        result = self.a.__mul__(self.b, layout=self.layout)
        assert result.layout == self.layout
        assert result.op == TensorOp.MUL
    
    def test_matmul_without_layout(self):
        """Test matrix multiplication without layout."""
        result = self.a @ self.b
        assert result.layout is None
        assert result.op == TensorOp.MATMUL
    
    def test_matmul_with_layout(self):
        """Test matrix multiplication with layout parameter."""
        result = self.a.matmul(self.b, layout=self.layout)
        assert result.layout == self.layout
        assert result.op == TensorOp.MATMUL
    
    def test_matmul_operator_with_layout(self):
        """Test @ operator with layout parameter."""
        result = self.a.__matmul__(self.b, layout=self.layout)
        assert result.layout == self.layout
        assert result.op == TensorOp.MATMUL
    
    def test_block_matmul_with_layout(self):
        """Test block matrix multiplication with layout."""
        result = self.a.block_matmul(self.b, layout=self.layout)
        assert result.layout == self.layout
        assert result.op == TensorOp.BLOCK_MATMUL
    
    def test_indexing_without_layout(self):
        """Test indexing without layout."""
        result = self.a[0]
        assert result.layout is None
        assert result.op == TensorOp.INDEX
    
    def test_indexing_with_layout(self):
        """Test indexing with layout parameter."""
        layout = "[0:1:1]"
        result = self.a.__getitem__(0, layout=layout)
        assert result.layout == layout
        assert result.op == TensorOp.INDEX


class TestTensorMethodsLayout:
    """Test tensor methods with layout parameters."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.a = TensorTerm.Tensor("a", [4, 4], True)
        self.layout = "[0:1:1]"
    
    def test_sum_without_layout(self):
        """Test sum without layout."""
        result = self.a.sum(0)
        assert result.layout is None
        assert result.op == TensorOp.SUM
    
    def test_sum_with_layout(self):
        """Test sum with layout parameter."""
        result = self.a.sum(0, layout=self.layout)
        assert result.layout == self.layout
        assert result.op == TensorOp.SUM
    
    def test_product_without_layout(self):
        """Test product without layout."""
        result = self.a.product(0)
        assert result.layout is None
        assert result.op == TensorOp.PRODUCT
    
    def test_product_with_layout(self):
        """Test product with layout parameter."""
        result = self.a.product(0, layout=self.layout)
        assert result.layout == self.layout
        assert result.op == TensorOp.PRODUCT
    
    def test_transpose_without_layout(self):
        """Test transpose without layout."""
        result = self.a.transpose()
        assert result.layout is None
        assert result.op == TensorOp.TRANSPOSE
    
    def test_transpose_with_layout(self):
        """Test transpose with layout parameter."""
        layout = "[1:4:1][0:4:1]"
        result = self.a.transpose(layout=layout)
        assert result.layout == layout
        assert result.op == TensorOp.TRANSPOSE
    
    def test_reshape_without_layout(self):
        """Test reshape without layout."""
        result = self.a.reshape(0, {0: 2, 1: 8})
        assert result.layout is None
        assert result.op == TensorOp.RESHAPE
    
    def test_reshape_with_layout(self):
        """Test reshape with layout parameter."""
        layout = "[0:2:1][1:8:1]"
        result = self.a.reshape(0, {0: 2, 1: 8}, layout=layout)
        assert result.layout == layout
        assert result.op == TensorOp.RESHAPE
    
    def test_permute_without_layout(self):
        """Test permute without layout."""
        result = self.a.permute({0: 1, 1: 0})
        assert result.layout is None
        assert result.op == TensorOp.PERMUTE
    
    def test_permute_with_layout(self):
        """Test permute with layout parameter."""
        layout = "[1:4:1][0:4:1]"
        result = self.a.permute({0: 1, 1: 0}, layout=layout)
        assert result.layout == layout
        assert result.op == TensorOp.PERMUTE
    
    def test_poly_without_layout(self):
        """Test Poly without layout."""
        result = self.a.Poly()
        assert result.layout is None
        assert result.op == TensorOp.POLY
    
    def test_poly_with_layout(self):
        """Test Poly with layout parameter."""
        layout = "[0:4:1][1:4:1]"
        result = self.a.Poly(layout=layout)
        assert result.layout == layout
        assert result.op == TensorOp.POLY


class TestStaticMethodsLayout:
    """Test static methods with layout parameters."""
    
    def test_conv2d_without_layout(self):
        """Test conv2d without layout."""
        a = TensorTerm.Tensor("a", [32, 32, 3], True)
        b = TensorTerm.Tensor("b", [3, 3, 3, 64], False)
        result = TensorTerm.conv2d(a, b, 1, "same")
        assert result.layout is None
        assert result.op == TensorOp.CONV2D
    
    def test_conv2d_with_layout(self):
        """Test conv2d with layout parameter."""
        a = TensorTerm.Tensor("a", [32, 32, 3], True)
        b = TensorTerm.Tensor("b", [3, 3, 3, 64], False)
        layout = "[0:32:1][1:32:1][2:64:1]"
        result = TensorTerm.conv2d(a, b, 1, "same", layout=layout)
        assert result.layout == layout
        assert result.op == TensorOp.CONV2D


class TestLayoutStringFormats:
    """Test various layout string formats."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.a = TensorTerm.Tensor("a", [4, 4], True)
        self.b = TensorTerm.Tensor("b", [4, 4], True)
    
    def test_row_major_layout(self):
        """Test row-major layout format."""
        layout = "[0:4:1][1:4:1]"
        result = self.a.matmul(self.b, layout=layout)
        assert result.layout == layout
    
    def test_column_major_layout(self):
        """Test column-major layout format."""
        layout = "[1:4:1][0:4:1]"
        result = self.a.matmul(self.b, layout=layout)
        assert result.layout == layout
    
    def test_layout_with_rolls(self):
        """Test layout format with rolls."""
        layout = "roll(0,1) [1:4:1][0:4:1]"
        result = self.a.matmul(self.b, layout=layout)
        assert result.layout == layout
    
    def test_3d_layout(self):
        """Test 3D tensor layout format."""
        a = TensorTerm.Tensor("a", [2, 4, 4], True)
        layout = "[0:2:1][1:4:1][2:4:1]"
        result = a.transpose(layout=layout)
        assert result.layout == layout
    
    def test_ciphertext_distribution_layout(self):
        """Test ciphertext distribution layout format."""
        layout = "[R:4:1];[0:4:1][1:4:1]"
        result = self.a.transpose(layout=layout)
        assert result.layout == layout
    
    def test_empty_layout_string(self):
        """Test that empty layout string is handled."""
        result = self.a.matmul(self.b, layout="")
        assert result.layout == ""
    
    def test_none_layout_parameter(self):
        """Test that None layout parameter is handled."""
        result = self.a.matmul(self.b, layout=None)
        assert result.layout is None


class TestLayoutComposition:
    """Test layout parameter in composed operations."""
    
    def test_chained_operations_with_layouts(self):
        """Test multiple operations with different layouts."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 4], True)
        
        # First operation with layout
        intermediate = a.matmul(b, layout="[0:4:1][1:4:1]")
        assert intermediate.layout == "[0:4:1][1:4:1]"
        
        # Second operation with different layout
        result = intermediate.transpose(layout="[1:4:1][0:4:1]")
        assert result.layout == "[1:4:1][0:4:1]"
        
        # Third operation without layout
        final = result.sum(0)
        assert final.layout is None
    
    def test_mixed_layout_and_no_layout_operations(self):
        """Test mixing operations with and without layouts."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 4], True)
        
        # Some operations with layout, some without
        step1 = a.matmul(b, layout="[0:4:1][1:4:1]")
        step2 = step1 + b  # No layout
        step3 = step2.transpose(layout="[1:4:1][0:4:1]")
        step4 = step3.sum(0)  # No layout
        
        assert step1.layout == "[0:4:1][1:4:1]"
        assert step2.layout is None
        assert step3.layout == "[1:4:1][0:4:1]"
        assert step4.layout is None


class TestLayoutEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_layout_with_very_long_string(self):
        """Test layout parameter with very long string."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 4], True)
        
        long_layout = "[0:4:1]" * 100  # Very long layout string
        result = a.matmul(b, layout=long_layout)
        assert result.layout == long_layout
    
    def test_layout_with_special_characters(self):
        """Test layout parameter with special characters."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 4], True)
        
        special_layout = "[0:4:1][1:4:1] !@#$%^&*()"
        result = a.matmul(b, layout=special_layout)
        assert result.layout == special_layout
    
    def test_layout_parameter_preservation_in_children(self):
        """Test that layout parameter doesn't affect child terms."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 4], True)
        
        # Create operation with layout
        result = a.matmul(b, layout="[0:4:1][1:4:1]")
        
        # Child terms should not have layout modified
        assert result.cs[0] == a
        assert result.cs[1] == b
        assert a.layout is None
        assert b.layout is None


class TestLayoutBackwardCompatibility:
    """Test that existing functionality still works."""
    
    def test_existing_operator_syntax_still_works(self):
        """Test that existing operator syntax continues to work."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 4], True)
        
        # Test all operator syntaxes
        add_result = a + b
        sub_result = a - b
        mul_result = a * b
        matmul_result = a @ b
        
        # All should work without layout
        assert add_result.layout is None
        assert sub_result.layout is None
        assert mul_result.layout is None
        assert matmul_result.layout is None
    
    def test_existing_method_syntax_still_works(self):
        """Test that existing method syntax continues to work."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        
        # Test all method syntaxes
        sum_result = a.sum(0)
        transpose_result = a.transpose()
        reshape_result = a.reshape(0, {0: 2, 1: 8})
        permute_result = a.permute({0: 1, 1: 0})
        
        # All should work without layout
        assert sum_result.layout is None
        assert transpose_result.layout is None
        assert reshape_result.layout is None
        assert permute_result.layout is None
    
    def test_property_syntax_still_works(self):
        """Test that property syntax (like .T) still works."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        
        transpose_result = a.T
        
        assert transpose_result.layout is None
        assert transpose_result.op == TensorOp.TRANSPOSE


if __name__ == "__main__":
    pytest.main([__file__])
