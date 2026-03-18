import numpy as np
from tqdm import tqdm

from frontends.tensor_args import TensorPlaceholderArgs
from ir.kernel import KernelOp
from util.layout_util import apply_layout


def serialize_mlp_mnist_inputs(kernel):
    from tests.mnist.test_mlp_mnist import (
        _load_mnist_test_set,
    )

    input_layout = None
    for k in kernel.post_order():
        if (
            k.op == KernelOp.TENSOR
            and TensorPlaceholderArgs.from_term(k.layout.term).name == "input"
        ):
            input_layout = k.layout
            break

    if input_layout is None:
        raise ValueError("Input layout not found")

    images, labels = _load_mnist_test_set()
    all_inputs = []
    for idx in tqdm(range(10000)):
        # Load a single MNIST test sample.
        assert images.shape[0] == labels.shape[0] and images.shape[0] > 0
        x = images[idx : idx + 1]  # [1, 1, 28, 28]

        # Flatten to [1, 784] to match the traced model's expected input.
        x_flat = x.view(1, -1).numpy()

        x_flat = np.array(apply_layout(x_flat, input_layout))
        all_inputs.append(x_flat)
    all_inputs = np.array(all_inputs)
    np.savez_compressed("heir/mlp_mnist_heir/mlp_mnist_inputs.npz", inputs=all_inputs)
