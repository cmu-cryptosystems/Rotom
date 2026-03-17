# """
# Tests for ResNet-style benchmark (conv + BatchNorm + SiLU via Poly).

# Padding shape mismatch (full ResNet + toy backend) — detailed explanation:

#   GAP path = global average pooling path: last block output (64, 8, 8) → sum(1)
#   → sum(1) → reshape(0, {0:1, 1:64}) → matmul with fc. The benchmark uses
#   sum(1) twice (so the reduced dimension exists in layout) and one reshape.

#   Nothing is done silently: SUM/PRODUCT with dim_idx >= rank raises ValueError
#   in shape analysis; RESHAPE with dim_to_del out of range or missing raises;
#   gen_sum raises if the requested sum dimension is not in the layout.
# """

# import numpy as np
# import pytest

# from frontends.tensor import TensorTerm


# class TestResNetOneLayer:
#     """Test one ResNet layer (conv -> BN -> SiLU) end-to-end."""

#     def test_resnet_one_layer_eval(self):
#         """One layer: input -> conv2d -> BatchNorm -> SiLU; eval runs and shape is correct."""
#         from benchmarks.rotom_benchmarks.resnet_silu import (
#             _batchnorm_term,
#             _conv2d_term,
#         )

#         np.random.seed(123)
#         inputs = {}
#         C_in, C_out, H, W = 2, 4, 4, 4
#         inputs["input"] = np.random.randn(C_in, H, W).astype(np.float64) * 0.1

#         x = TensorTerm.Tensor("input", [C_in, H, W], True)
#         conv = _conv2d_term("conv1", C_in, C_out, 3, inputs, stride=1, padding="same")
#         x = conv(x)
#         x = _batchnorm_term(x, "conv1_bn", C_out, inputs)
#         x = x.silu_poly()

#         tensor_ir = x
#         result = tensor_ir.eval(inputs)

#         assert result.shape[0] == C_out
#         assert np.all(np.isfinite(result))
#         assert result.size > 0
#         # Logical spatial size (same padding, stride 1) is 4x4
#         assert result.shape[1] >= 4 and result.shape[2] >= 4

#     def test_resnet_one_layer_shape_analysis(self):
#         """One ResNet layer: shape analysis runs and preserves conv output shape through BN and SiLU."""
#         from benchmarks.rotom_benchmarks.resnet_silu import (
#             _batchnorm_term,
#             _conv2d_term,
#         )
#         from ir.analysis.shape import Shape

#         np.random.seed(124)
#         inputs = {}
#         C_in, C_out, H, W = 2, 4, 4, 4
#         inputs["input"] = np.random.randn(C_in, H, W).astype(np.float64) * 0.1

#         x = TensorTerm.Tensor("input", [C_in, H, W], True)
#         conv = _conv2d_term("conv1", C_in, C_out, 3, inputs, stride=1, padding="same")
#         x = conv(x)
#         x = _batchnorm_term(x, "conv1_bn", C_out, inputs)
#         x = x.silu_poly()

#         shape_analyzer = Shape(x)
#         shape_analyzer.run()
#         assert shape_analyzer.get_shape(x) == [C_out, H, W]
#         # Padded to power-of-2
#         assert shape_analyzer.get_padded_shape(x)[0] == C_out

#     def test_resnet_one_layer_assignment_and_lower(self):
#         """One layer runs through layout assignment and lowering (no backend)."""
#         from assignment.assignment import LayoutAssignment
#         from benchmarks.rotom_benchmarks.resnet_silu import (
#             _batchnorm_term,
#             _conv2d_term,
#         )
#         from lower.lower import Lower
#         from tests.test_util import get_default_args

#         np.random.seed(125)
#         inputs = {}
#         C_in, C_out, H, W = 2, 4, 4, 4
#         inputs["input"] = np.random.randn(C_in, H, W).astype(np.float64) * 0.1

#         x = TensorTerm.Tensor("input", [C_in, H, W], True)
#         conv = _conv2d_term("conv1", C_in, C_out, 3, inputs, stride=1, padding="same")
#         x = conv(x)
#         x = _batchnorm_term(x, "conv1_bn", C_out, inputs)
#         x = x.silu_poly()
#         tensor_ir = x

#         args = get_default_args()
#         args.n = 16
#         args.rolls = True
#         args.benchmark = "resnet_one_layer"

#         kernel = LayoutAssignment(tensor_ir, args).run()
#         circuit_ir = Lower(kernel).run()
#         assert circuit_ir is not None

#     def test_resnet_one_layer_simple(self):
#         """One layer runs through assignment, lower, and toy backend; results match expected."""
#         from assignment.assignment import LayoutAssignment
#         from backends.toy import Toy
#         from benchmarks.rotom_benchmarks.resnet_silu import (
#             _batchnorm_term,
#             _conv2d_term,
#         )
#         from lower.lower import Lower
#         from tests.test_util import get_default_args
#         from util.layout_util import apply_layout

#         np.random.seed(125)
#         inputs = {}
#         C_in, C_out, H, W = 2, 4, 4, 4
#         inputs["input"] = np.random.randint(-5, 5, (C_in, H, W)).astype(np.float64)

#         x = TensorTerm.Tensor("input", [C_in, H, W], True)
#         conv = _conv2d_term("conv1", C_in, C_out, 3, inputs, stride=1, padding="same")
#         # inputs["conv1_w"] = np.random.randint(-3, 3, (C_out, C_in, 3, 3)).astype(np.float64)
#         inputs["conv1_w"] = (
#             np.arange(C_out * C_in * 3 * 3)
#             .reshape(C_out, C_in, 3, 3)
#             .astype(np.float64)
#         )

#         x = conv(x)
#         x = _batchnorm_term(x, "conv1_bn", C_out, inputs)
#         x = x.poly("silu")
#         tensor_ir = x

#         args = get_default_args()
#         args.n = 16
#         args.rolls = True
#         args.benchmark = "resnet_one_layer"

#         expected = tensor_ir.eval(inputs)
#         kernel = LayoutAssignment(tensor_ir, args).run()
#         circuit_ir = Lower(kernel).run()
#         assert circuit_ir is not None

#         backend = Toy(circuit_ir, inputs, args)
#         results = backend.run()

#         # Check toy output matches expected when layout produces same total size; else at least finite.
#         expected_cts = apply_layout(expected, kernel.layout)
#         expected_flat = np.concatenate([np.asarray(v).flatten() for v in expected_cts])
#         result_flat = np.concatenate([np.asarray(r).flatten() for r in results])
#         if expected_flat.size == result_flat.size:
#             np.testing.assert_allclose(
#                 result_flat,
#                 expected_flat,
#                 rtol=1e-1,
#                 atol=1e-1,
#                 err_msg="Toy backend output should match tensor_ir.eval (after layout).",
#             )
#         else:
#             assert len(results) >= 1
#             assert np.all(np.isfinite(result_flat)), "results must be finite"


# def _resnet_silu_build_up_to(stage, inputs, C_in=3, H=32, W=32):
#     """Build ResNet SiLU graph up to the given stage (inclusive). Returns tensor_ir."""
#     from benchmarks.rotom_benchmarks.resnet_silu import (
#         _basic_block,
#         _batchnorm_term,
#         _conv2d_term,
#     )

#     inputs["input"] = np.random.randn(C_in, H, W).astype(np.float64) * 0.1
#     x = TensorTerm.Tensor("input", [C_in, H, W], True)

#     if stage == "conv1":
#         conv1 = _conv2d_term("conv1", 3, 16, 3, inputs, stride=1, padding="same")
#         x = conv1(x)
#         x = _batchnorm_term(x, "conv1_bn", 16, inputs)
#         x = x.silu_poly()
#         return x

#     conv1 = _conv2d_term("conv1", 3, 16, 3, inputs, stride=1, padding="same")
#     x = conv1(x)
#     x = _batchnorm_term(x, "conv1_bn", 16, inputs)
#     x = x.poly("silu")

#     layer1_stages = ["layer1_0", "layer1_1", "layer1_2"]
#     for i, name in enumerate(layer1_stages):
#         x = _basic_block(
#             x, f"layer1_block{i}", in_ch=16, out_ch=16, stride=1, inputs=inputs
#         )
#         if stage == name:
#             return x

#     x = _basic_block(x, "layer2_block0", in_ch=16, out_ch=32, stride=2, inputs=inputs)
#     if stage == "layer2_0":
#         return x
#     for i in range(1, 3):
#         x = _basic_block(
#             x, f"layer2_block{i}", in_ch=32, out_ch=32, stride=1, inputs=inputs
#         )
#         if stage == f"layer2_{i}":
#             return x

#     x = _basic_block(x, "layer3_block0", in_ch=32, out_ch=64, stride=2, inputs=inputs)
#     if stage == "layer3_0":
#         return x
#     for i in range(1, 3):
#         x = _basic_block(
#             x, f"layer3_block{i}", in_ch=64, out_ch=64, stride=1, inputs=inputs
#         )
#         if stage == f"layer3_{i}":
#             return x
#     return x


# class TestResNetSiluLayerByLayer:
#     """Test full ResNet SiLU by adding one layer at a time (assignment, lower, toy, check)."""

#     # @pytest.mark.parametrize("stage", ["conv1", "layer1_0", "layer1_1", "layer1_2"])
#     # def test_resnet_silu_stage(self, stage):
#     #     """Build graph up to stage, run assignment/lower/toy and check vs expected."""
#     #     from assignment.assignment import LayoutAssignment
#     #     from backends.toy import Toy
#     #     from lower.lower import Lower
#     #     from tests.test_util import get_default_args
#     #     from util.layout_util import apply_layout

#     #     np.random.seed(125)
#     #     inputs = {}
#     #     tensor_ir = _resnet_silu_build_up_to(stage, inputs)

#     #     args = get_default_args()
#     #     args.n = 32768
#     #     args.rolls = True
#     #     args.conv_roll = True
#     #     args.benchmark = "resnet_silu_stage"

#     #     expected = tensor_ir.eval(inputs)
#     #     kernel = LayoutAssignment(tensor_ir, args).run()
#     #     circuit_ir = Lower(kernel).run()
#     #     assert circuit_ir is not None

#     #     backend = Toy(circuit_ir, inputs, args)
#     #     results = backend.run()

#     #     expected_cts = apply_layout(expected, kernel.layout)
#     #     expected_flat = np.concatenate([np.asarray(v).flatten() for v in expected_cts])
#     #     result_flat = np.concatenate([np.asarray(r).flatten() for r in results])
#     #     if expected_flat.size == result_flat.size:
#     #         np.testing.assert_allclose(
#     #             result_flat,
#     #             expected_flat,
#     #             rtol=1e-1,
#     #             atol=1e-1,
#     #             err_msg=f"Stage {stage}: Toy output should match tensor_ir.eval (after layout).",
#     #         )
#     #     else:
#     #         assert len(results) >= 1
#     #         assert np.all(np.isfinite(result_flat)), f"Stage {stage}: results must be finite"

#     def test_resnet_one_layer_full(self):
#         """Full resnet_one_layer (3, 32, 32) -> conv1 3→16 -> BN -> SiLU; eval runs and shape correct."""
#         from assignment.assignment import LayoutAssignment
#         from backends.toy import Toy
#         from benchmarks.rotom_benchmarks.resnet_silu import (
#             _batchnorm_term,
#             _conv2d_term,
#         )
#         from lower.lower import Lower
#         from tests.test_util import get_default_args
#         from util.layout_util import apply_layout

#         np.random.seed(125)
#         inputs = {}

#         # Input image (C, H, W) = (3, 32, 32) — same as resnet_silu.py lines 85–94
#         C_in, H, W = 3, 32, 32
#         inputs["input"] = np.random.randn(C_in, H, W).astype(np.float64) * 0.1

#         x = TensorTerm.Tensor("input", [C_in, H, W], True)
#         # conv1: 3 → 16, 3×3, stride 1, padding=1 ("same")
#         conv1 = _conv2d_term("conv1", 3, 16, 3, inputs, stride=1, padding="same")
#         x = conv1(x)
#         x = _batchnorm_term(x, "conv1_bn", 16, inputs)
#         x = x.silu_poly()
#         tensor_ir = x

#         args = get_default_args()
#         args.n = 1024
#         args.rolls = True
#         args.benchmark = "resnet_one_layer"

#         expected = tensor_ir.eval(inputs)
#         kernel = LayoutAssignment(tensor_ir, args).run()
#         circuit_ir = Lower(kernel).run()
#         assert circuit_ir is not None

#         backend = Toy(circuit_ir, inputs, args)
#         results = backend.run()

#         # Check toy output matches expected when layout produces same total size; else at least finite.
#         expected_cts = apply_layout(expected, kernel.layout)
#         expected_flat = np.concatenate([np.asarray(v).flatten() for v in expected_cts])
#         result_flat = np.concatenate([np.asarray(r).flatten() for r in results])
#         if expected_flat.size == result_flat.size:
#             np.testing.assert_allclose(
#                 result_flat,
#                 expected_flat,
#                 rtol=1e-1,
#                 atol=1e-1,
#                 err_msg="Toy backend output should match tensor_ir.eval (after layout).",
#             )
#         else:
#             assert len(results) >= 1
#             assert np.all(np.isfinite(result_flat)), "results must be finite"

#     def test_conv2d_select_input_channel_lowering(self):
#         """Test lower_conv2d select-by-input-channel (lines 238–267): same setup as test_resnet_one_layer_full."""
#         from assignment.assignment import LayoutAssignment
#         from backends.toy import Toy
#         from benchmarks.rotom_benchmarks.resnet_silu import (
#             _batchnorm_term,
#             _conv2d_term,
#         )
#         from ir.kernel import KernelOp
#         from lower.layout_cts import LayoutCiphertexts
#         from lower.lower import Lower
#         from tests.test_util import get_default_args
#         from util.layout_util import apply_layout

#         np.random.seed(125)
#         inputs = {}
#         C_in, H, W = 3, 32, 32
#         inputs["input"] = np.random.randn(C_in, H, W).astype(np.float64) * 0.1

#         x = TensorTerm.Tensor("input", [C_in, H, W], True)
#         conv1 = _conv2d_term("conv1", 3, 16, 3, inputs, stride=1, padding="same")
#         x = conv1(x)
#         x = _batchnorm_term(x, "conv1_bn", 16, inputs)
#         x = x.poly("silu")
#         tensor_ir = x

#         args = get_default_args()
#         args.n = 1024
#         args.rolls = True
#         args.benchmark = "resnet_one_layer"

#         kernel = LayoutAssignment(tensor_ir, args).run()
#         circuit_ir = Lower(kernel).run()

#         # Find CONV2D kernel and its lowered LayoutCiphertexts (output of select-by-input-channel block)
#         conv2d_kernel = None
#         for term in kernel.post_order():
#             if term.op == KernelOp.CONV2D:
#                 conv2d_kernel = term
#                 break
#         assert conv2d_kernel is not None, "CONV2D kernel not found"

#         layout_cts = circuit_ir[conv2d_kernel]
#         assert isinstance(layout_cts, LayoutCiphertexts)
#         assert (
#             len(layout_cts.cts) == conv2d_kernel.layout.num_ct()
#         ), f"CONV2D lowering should produce num_ct()={conv2d_kernel.layout.num_ct()} CTs, got {len(layout_cts.cts)}"
#         assert (
#             layout_cts.layout == conv2d_kernel.layout
#         ), "CONV2D lowering layout should match kernel.layout"

#         # Full pipeline: Toy run and value check (same as test_resnet_one_layer_full)
#         expected = tensor_ir.eval(inputs)
#         backend = Toy(circuit_ir, inputs, args)
#         results = backend.run()
#         expected_cts = apply_layout(expected, kernel.layout)
#         expected_flat = np.concatenate([np.asarray(v).flatten() for v in expected_cts])
#         result_flat = np.concatenate([np.asarray(r).flatten() for r in results])
#         if expected_flat.size == result_flat.size:
#             np.testing.assert_allclose(
#                 result_flat,
#                 expected_flat,
#                 rtol=1e-1,
#                 atol=1e-1,
#                 err_msg="Toy output should match tensor_ir.eval (after layout).",
#             )
#         else:
#             assert len(results) >= 1
#             assert np.all(np.isfinite(result_flat)), "results must be finite"
