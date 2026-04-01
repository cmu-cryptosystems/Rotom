"""
ResNet20 benchmark.
"""

import numpy as np

from frontends.tensor import TensorTerm


def _silu_poly_call(x: TensorTerm) -> TensorTerm:
    """Same ``PolyCall`` as CIFAR ResNet tensor IR (see ``resnet20_tensor_ir._silu_poly``)."""
    return x.poly_call("silu", -20.0, 20.0)


def _conv2d_term(name, in_ch, out_ch, k, inputs, stride=1, padding="same"):
    """Create conv2d term and register its weights in inputs."""
    w_name = f"{name}_w"
    if w_name not in inputs:
        inputs[w_name] = np.random.randn(out_ch, in_ch, k, k).astype(np.float64) * 0.1
    w_term = TensorTerm.Tensor(w_name, [out_ch, in_ch, k, k], False)
    return lambda x: TensorTerm.conv2d(x, w_term, stride, padding)


def _batchnorm_term(x, prefix, num_ch, inputs, eps=1e-5):
    """Register BatchNorm params in inputs and return Poly(BN) term."""
    inputs[f"{prefix}_mean"] = np.random.randn(num_ch).astype(np.float64) * 0.1
    inputs[f"{prefix}_var"] = np.abs(np.random.randn(num_ch).astype(np.float64)) + 0.1
    inputs[f"{prefix}_gamma"] = np.ones(num_ch, dtype=np.float64)
    inputs[f"{prefix}_beta"] = np.zeros(num_ch, dtype=np.float64)
    return TensorTerm.batchnorm(
        x,
        f"{prefix}_mean",
        f"{prefix}_var",
        f"{prefix}_gamma",
        f"{prefix}_beta",
        eps=eps,
    )


def _basic_block(x, name, in_ch, out_ch, stride, inputs):
    """
    ResNet BasicBlock: conv-BN-SiLU, conv-BN, (+ shortcut), SiLU.
    """
    conv1 = _conv2d_term(
        f"{name}_conv1", in_ch, out_ch, 3, inputs, stride=stride, padding="same"
    )
    conv2 = _conv2d_term(
        f"{name}_conv2", out_ch, out_ch, 3, inputs, stride=1, padding="same"
    )

    out = conv1(x)
    out = _batchnorm_term(out, f"{name}_conv1_bn", out_ch, inputs)
    out = _silu_poly_call(out)
    out = conv2(out)
    out = _batchnorm_term(out, f"{name}_conv2_bn", out_ch, inputs)

    if stride != 1 or in_ch != out_ch:
        shortcut_conv = _conv2d_term(
            f"{name}_shortcut", in_ch, out_ch, 1, inputs, stride=stride, padding="same"
        )
        shortcut = shortcut_conv(x)
        shortcut = _batchnorm_term(shortcut, f"{name}_shortcut_bn", out_ch, inputs)
    else:
        shortcut = x

    out = out + shortcut
    out = _silu_poly_call(out)
    return out


def resnet_silu():
    """
    Conv2d-based ResNet20-style benchmark with BatchNorm and SiLU via Poly.

    Shapes (per image, no batch dim in the tensor IR):
      - Input:   [3, 32, 32]
      - conv1:   [16, 32, 32]
      - layer1:  3 blocks, each [16, 32, 32]
      - layer2:  3 blocks, first does downsample to [32, 16, 16]
      - layer3:  3 blocks, first does downsample to [64, 8, 8]
      - GAP:     sum over H and W → [64]
      - FC:      [64 → 10] → logits [1, 10]

    Returns:
        (tensor_ir, inputs, n) for use with main.py --benchmark resnet_silu.
        n is set to 2**16, matching DaCapo's nt.
    """
    inputs = {}

    # Input image (C, H, W) = (3, 32, 32)
    C_in, H, W = 3, 32, 32
    inputs["input"] = np.random.randn(C_in, H, W) * 0.1
    x = TensorTerm.Tensor("input", [C_in, H, W], True)

    # conv1: 3 → 16, 3×3, stride 1, padding=1 ("same")
    conv1 = _conv2d_term("conv1", 3, 16, 3, inputs, stride=1, padding="same")
    x = conv1(x)
    x = _batchnorm_term(x, "conv1_bn", 16, inputs)
    x = _silu_poly_call(x)

    # layer1: 3 blocks, 16 channels, stride 1
    for i in range(3):
        x = _basic_block(
            x, f"layer1_block{i}", in_ch=16, out_ch=16, stride=1, inputs=inputs
        )

    # layer2: 3 blocks, 32 channels; first block stride 2 (downsample)
    x = _basic_block(x, "layer2_block0", in_ch=16, out_ch=32, stride=2, inputs=inputs)
    for i in range(1, 3):
        x = _basic_block(
            x, f"layer2_block{i}", in_ch=32, out_ch=32, stride=1, inputs=inputs
        )

    # layer3: 3 blocks, 64 channels; first block stride 2 (downsample)
    x = _basic_block(x, "layer3_block0", in_ch=32, out_ch=64, stride=2, inputs=inputs)
    for i in range(1, 3):
        x = _basic_block(
            x, f"layer3_block{i}", in_ch=64, out_ch=64, stride=1, inputs=inputs
        )

    # Global average pool: sum over H (dim 1) then W (dim 1 after first sum).
    # Use sum(1) twice so shape analysis (no keepdims) and assignment layout agree.
    x = x.sum(1)  # (64, 8, 8) -> (64, 8)
    x = x.sum(1)  # (64, 8) -> (64,)

    # Reshape to [1, 64] for final FC
    x = x.reshape(0, {0: 1, 1: 64})  # (64,) -> (1, 64)

    # Final linear: 64 → 10
    num_classes = 10
    inputs["fc"] = np.random.randn(64, num_classes) * 0.1
    fc = TensorTerm.Tensor("fc", [64, num_classes], False)
    tensor_ir = x @ fc  # [1, 10]

    n = 2**16
    return tensor_ir, inputs, n


def resnet_silu_one_layer():
    """
    One ResNet layer (conv -> BN -> SiLU) with ResNet first-layer channel dims (3→16).
    Spatial size 8×8 so assignment/lowering complete (32×32 can fail to produce conv2d candidates).

    Returns:
        (tensor_ir, inputs, n) for use with main.py --benchmark resnet_silu_one_layer.
    """
    np.random.seed(125)
    inputs = {}
    C_in, C_out, H, W = 3, 16, 8, 8
    inputs["input"] = np.random.randn(C_in, H, W).astype(np.float64) * 0.1
    x = TensorTerm.Tensor("input", [C_in, H, W], True)
    conv = _conv2d_term("conv1", C_in, C_out, 3, inputs, stride=1, padding="same")
    x = conv(x)
    x = _batchnorm_term(x, "conv1_bn", C_out, inputs)
    x = _silu_poly_call(x)
    tensor_ir = x
    n = 4096
    return tensor_ir, inputs, n
