"""TensorTerm evaluation helpers.

This module contains `TensorEvaluator`, which implements the concrete
evaluation semantics for the Tensor frontend IR defined in `tensor.py`.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


class TensorEvaluator:
    """Evaluate TensorTerm graphs.

    This class implements the concrete evaluation semantics for the Tensor IR.
    Keeping evaluation logic separate from `TensorTerm` makes it easier to:
      - add new ops without bloating the IR node
      - plug in alternative evaluators (e.g. debug vs optimized numpy)
    """

    @staticmethod
    def _round_to_ceiling_power_of_2(n: int) -> int:
        if n <= 0:
            raise ValueError("Input must be a positive number.")
        return 1 if n == 1 else 2 ** math.ceil(math.log2(n))

    @staticmethod
    def _eval_conv2d(
        input_tensor: np.ndarray,
        filter_tensor: np.ndarray,
        stride: int,
        padding: str,
    ) -> np.ndarray:
        input_shape = input_tensor.shape
        filter_shape = filter_tensor.shape
        if padding == "valid":
            h_o = (input_shape[1] - filter_shape[2]) // stride + 1
            w_o = (input_shape[2] - filter_shape[3]) // stride + 1
            output_shape = [filter_shape[0], h_o, w_o]
        elif padding == "same":
            if stride == 1:
                output_shape = [filter_shape[0], input_shape[1], input_shape[2]]
            else:
                p_h = filter_shape[2] // 2
                p_w = filter_shape[3] // 2
                h_pad = input_shape[1] + 2 * p_h
                w_pad = input_shape[2] + 2 * p_w
                h_o = (h_pad - filter_shape[2]) // stride + 1
                w_o = (w_pad - filter_shape[3]) // stride + 1
                output_shape = [filter_shape[0], h_o, w_o]
        else:
            raise ValueError(f"Unsupported padding mode: {padding!r}")

        if padding == "same":
            if stride == 1:
                pad_top = max(
                    0,
                    math.floor(
                        (
                            stride * (output_shape[1] - 1)
                            - input_shape[1]
                            + filter_shape[2]
                        )
                        / 2
                    ),
                )
                pad_bot = max(
                    0,
                    math.ceil(
                        (
                            stride * (output_shape[1] - 1)
                            - input_shape[1]
                            + filter_shape[2]
                        )
                        / 2
                    ),
                )
                pad_left = max(
                    0,
                    math.floor(
                        (
                            stride * (output_shape[2] - 1)
                            - input_shape[2]
                            + filter_shape[3]
                        )
                        / 2
                    ),
                )
                pad_right = max(
                    0,
                    math.ceil(
                        (
                            stride * (output_shape[2] - 1)
                            - input_shape[2]
                            + filter_shape[3]
                        )
                        / 2
                    ),
                )
            else:
                p_h = filter_shape[2] // 2
                p_w = filter_shape[3] // 2
                pad_top = pad_bot = p_h
                pad_left = pad_right = p_w

            padded_input_tensor = []
            for channel_tensor in input_tensor:
                padded_input_tensor.append(
                    np.pad(
                        channel_tensor,
                        pad_width=((pad_top, pad_bot), (pad_left, pad_right)),
                        mode="constant",
                        constant_values=0,
                    )
                )
            input_tensor = padded_input_tensor

        output_tensor = np.zeros(output_shape)
        for in_c in range(input_shape[0]):
            for out_c in range(output_shape[0]):
                for i in range(output_shape[1]):
                    for j in range(output_shape[2]):
                        i_start = i * stride
                        j_start = j * stride
                        i_end = i_start + filter_shape[2]
                        j_end = j_start + filter_shape[3]

                        patch = []
                        for x in range(i_start, i_end):
                            row = []
                            for y in range(j_start, j_end):
                                row.append(input_tensor[in_c][x][y])
                            patch.append(row)
                        patch = np.array(patch)

                        f_in_idx = min(in_c, filter_shape[1] - 1)
                        output_tensor[out_c][i][j] += np.sum(
                            patch * filter_tensor[out_c][f_in_idx]
                        )

        return output_tensor

    def _eval_poly(
        self, x: np.ndarray, func: Any, inputs: Dict[str, Any]
    ) -> np.ndarray:
        if callable(func):
            return np.asarray(func(x))
        if func == "identity":
            return x
        # Allow either POLY("relu_exact") or POLY_CALL("relu", ...) to be treated
        # as ReLU at eval time. The latter is primarily for layout / backend control.
        if func == "relu_exact" or func == "relu":
            return np.maximum(x, 0.0)
        if func == "silu":
            # Plaintext exact SiLU for PolyCall("silu", ...).
            x = np.asarray(x, dtype=np.float64)
            return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))
        if isinstance(func, tuple) and len(func) >= 5 and func[0] == "batchnorm":
            _, mean_key, var_key, gamma_key, beta_key = func[:5]
            eps = float(func[5]) if len(func) > 5 else 1e-5
            mean = np.asarray(inputs[mean_key])
            var = np.asarray(inputs[var_key])
            gamma = np.asarray(inputs[gamma_key])
            beta = np.asarray(inputs[beta_key])
            if x.ndim >= 3:
                ch_dim = x.shape[0]
                bc_shape = (ch_dim,) + (1,) * (x.ndim - 1)
            elif x.ndim == 2:
                ch_dim = x.shape[-1]
                bc_shape = (1, ch_dim)
            else:
                ch_dim = mean.size
                bc_shape = None
            if ch_dim > mean.size:
                pad_len = ch_dim - mean.size
                mean = np.concatenate([mean, np.zeros(pad_len, dtype=mean.dtype)])
                var = np.concatenate([var, np.ones(pad_len, dtype=var.dtype)])
                gamma = np.concatenate([gamma, np.ones(pad_len, dtype=gamma.dtype)])
                beta = np.concatenate([beta, np.zeros(pad_len, dtype=beta.dtype)])
            if bc_shape is not None:
                mean = mean.reshape(bc_shape)
                var = var.reshape(bc_shape)
                gamma = gamma.reshape(bc_shape)
                beta = beta.reshape(bc_shape)
            inv_std = 1.0 / np.sqrt(var + eps)
            return gamma * (x - mean) * inv_std + beta
        if isinstance(func, (list, tuple)) and len(func) > 0:
            try:
                coeffs = [float(c) for c in func]
            except (TypeError, ValueError):
                coeffs = None
            if coeffs is not None:
                out = np.zeros_like(x, dtype=np.float64)
                for i, c in enumerate(coeffs):
                    out = out + c * (x.astype(np.float64) ** i)
                return out
        raise NotImplementedError(f"Poly func {func!r} not implemented for eval")

    def eval_term(self, term: Any, env: Dict[Any, Any], inputs: Dict[str, Any]) -> Any:
        op = getattr(term, "op")
        op_name = getattr(op, "value", op)
        match op_name:
            case "Tensor":
                shape = inputs[term.cs[0]].shape
                rounded_shape = [self._round_to_ceiling_power_of_2(s) for s in shape]
                padding = [0] * len(shape)
                for i, (a, b) in enumerate(zip(shape, rounded_shape)):
                    padding[i] = b - a

                if len(padding) == 1:
                    padded_tensor = np.pad(
                        inputs[term.cs[0]],
                        pad_width=((0, padding[0])),
                        mode="constant",
                        constant_values=0,
                    )
                elif len(padding) == 2:
                    padded_tensor = np.pad(
                        inputs[term.cs[0]],
                        pad_width=((0, padding[0]), (0, padding[1])),
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    return np.array(inputs[term.cs[0]])
                return np.array(padded_tensor)
            case "Add":
                return env[term.cs[0]] + env[term.cs[1]]
            case "Sub":
                return env[term.cs[0]] - env[term.cs[1]]
            case "Mul":
                return env[term.cs[0]] * env[term.cs[1]]
            case "Sum":
                return np.sum(env[term.cs[0]], axis=term.cs[1], keepdims=False)
            case "MatMul":
                return env[term.cs[0]] @ env[term.cs[1]]
            case "Transpose":
                return env[term.cs[0]].T
            case "Conv":
                from .tensor_args import Conv2dArgs

                args = Conv2dArgs.from_term(term)
                return self._eval_conv2d(
                    env[args.input], env[args.filter], args.stride, args.padding
                )
            case "Const":
                return term.cs[0]
            case "Index":

                def _to_slice(obj):
                    if isinstance(obj, slice) or obj is Ellipsis or obj is None:
                        return obj
                    if isinstance(obj, list) and len(obj) in (2, 3):
                        return slice(*obj)
                    if isinstance(obj, dict) and ("start" in obj or "stop" in obj):
                        return slice(obj.get("start"), obj.get("stop"), obj.get("step"))
                    return obj

                item = term.cs[1]
                if isinstance(item, tuple):
                    item = tuple(_to_slice(x) for x in item)
                else:
                    item = _to_slice(item)

                return env[term.cs[0]][item]
            case "Reshape":
                from .tensor_args import ReshapeArgs

                args = ReshapeArgs.from_term(term)
                tensor = env[args.input]
                shape = {}
                for i, s in enumerate(tensor.shape):
                    shape[i] = s
                del shape[args.dim]
                for k, v in args.shape.items():
                    shape[k] = self._round_to_ceiling_power_of_2(v)
                shape_list = [shape[k] for k in sorted(shape.keys())]
                return tensor.reshape(shape_list)
            case "Permute":
                tensor = env[term.cs[0]]
                return np.moveaxis(tensor, term.cs[1].keys(), term.cs[1].values())
            case "Rescale":
                scale_value = 2 ** term.cs[1]
                return env[term.cs[0]] / scale_value
            case "PolyCall":
                x = env[term.cs[0]]
                if len(term.cs) >= 4:
                    from .tensor_args import PolyCallArgs

                    func = PolyCallArgs.from_term(term).name
                else:
                    func = term.cs[1] if len(term.cs) > 1 else "identity"
                return self._eval_poly(x, func, inputs)
            case _:
                raise NotImplementedError(op_name)

    def eval(self, term: Any, inputs: Dict[str, Any]) -> Any:
        env: Dict[Any, Any] = {}
        last = None
        for t in term.post_order():
            env[t] = self.eval_term(t, env, inputs)
            last = t
        assert last is not None
        return env[last]
