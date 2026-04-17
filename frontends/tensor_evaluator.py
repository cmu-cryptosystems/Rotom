"""TensorTerm evaluation helpers.

This module contains `TensorEvaluator`, which implements the concrete
evaluation semantics for the Tensor frontend IR defined in `tensor.py`.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from util.silu_polycall_eval import eval_silu_polycall


class TensorEvaluator:
    """Evaluate TensorTerm graphs.

    This class implements the concrete evaluation semantics for the Tensor IR.
    Keeping evaluation logic separate from `TensorTerm` makes it easier to:
      - add new ops without bloating the IR node
      - plug in alternative evaluators (e.g. debug vs optimized numpy)
    """

    @staticmethod
    def _trim_to_declared_shape(x: np.ndarray, term: Any) -> np.ndarray:
        """Trim evaluator output to the declared placeholder shape when available."""
        src = term.cs[0] if hasattr(term, "cs") and term.cs else None
        if src is not None and getattr(src, "op", None) is not None:
            src_op_name = getattr(getattr(src, "op"), "value", getattr(src, "op"))
            if src_op_name == "Tensor":
                declared = tuple(int(v) for v in src.cs[1])
                if x.ndim == len(declared):
                    slices = tuple(slice(0, d) for d in declared)
                    return np.asarray(x)[slices]
        return x

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
        groups: Any = 1,
    ) -> np.ndarray:
        input_shape = input_tensor.shape
        filter_shape = filter_tensor.shape
        c_in = int(input_shape[0])

        # Support TFLite-style depthwise filter [1, Kh, Kw, C_in*mult] in addition to
        # Rotom-style filter [C_out, C_per_group, Kh, Kw].
        if len(filter_shape) == 4 and filter_shape[0] == 1 and groups == "depthwise":
            mult = int(filter_shape[3]) // c_in
            if mult <= 0 or (mult * c_in) != int(filter_shape[3]):
                raise ValueError(
                    f"Invalid depthwise filter shape {filter_shape} for input channels {c_in}"
                )
            kh, kw = int(filter_shape[1]), int(filter_shape[2])
            filt = np.transpose(filter_tensor, (3, 0, 1, 2)).reshape(
                c_in * mult, 1, kh, kw
            )
            filter_tensor = filt
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

        c_out = int(output_shape[0])
        if groups == "depthwise":
            groups = c_in
        if not isinstance(groups, int) or groups <= 0:
            raise ValueError(
                f"groups must be positive int or 'depthwise', got {groups!r}"
            )
        if c_in % groups != 0:
            raise ValueError(f"input channels {c_in} not divisible by groups {groups}")
        if c_out % groups != 0:
            raise ValueError(
                f"output channels {c_out} not divisible by groups {groups}"
            )
        c_in_per_group = c_in // groups
        c_out_per_group = c_out // groups
        if int(filter_shape[1]) != c_in_per_group:
            raise ValueError(
                f"filter C_in/group mismatch: expected {c_in_per_group}, got {filter_shape[1]}"
            )

        output_tensor = np.zeros(output_shape)
        for out_c in range(c_out):
            group_idx = out_c // c_out_per_group
            in_start = group_idx * c_in_per_group
            in_end = in_start + c_in_per_group
            for in_c_idx in range(in_start, in_end):
                f_in_idx = in_c_idx - in_start
                for i in range(output_shape[1]):
                    for j in range(output_shape[2]):
                        i_start = i * stride
                        j_start = j * stride
                        i_end = i_start + filter_shape[2]
                        j_end = j_start + filter_shape[3]
                        patch = input_tensor[in_c_idx][i_start:i_end, j_start:j_end]
                        output_tensor[out_c][i][j] += np.sum(
                            patch * filter_tensor[out_c][f_in_idx]
                        )

        return output_tensor

    @staticmethod
    def _eval_avg_pool2d(
        input_tensor: np.ndarray, kernel: int, stride: int, padding: str
    ) -> np.ndarray:
        c, h, w = input_tensor.shape
        if padding == "valid":
            h_o = (h - kernel) // stride + 1
            w_o = (w - kernel) // stride + 1
            pad_top = pad_bottom = pad_left = pad_right = 0
        elif padding == "same":
            if stride == 1:
                h_o, w_o = h, w
                total_ph = max(0, (h_o - 1) * stride + kernel - h)
                total_pw = max(0, (w_o - 1) * stride + kernel - w)
                pad_top = total_ph // 2
                pad_bottom = total_ph - pad_top
                pad_left = total_pw // 2
                pad_right = total_pw - pad_left
            else:
                p = kernel // 2
                pad_top = pad_bottom = pad_left = pad_right = p
                h_o = (h + 2 * p - kernel) // stride + 1
                w_o = (w + 2 * p - kernel) // stride + 1
        else:
            raise ValueError(f"Unsupported padding mode: {padding!r}")

        if any(x != 0 for x in (pad_top, pad_bottom, pad_left, pad_right)):
            padded = np.zeros((c, h + pad_top + pad_bottom, w + pad_left + pad_right))
            padded[:, pad_top : pad_top + h, pad_left : pad_left + w] = input_tensor
            input_tensor = padded

        out = np.zeros((c, h_o, w_o))
        denom = float(kernel * kernel)
        for ch in range(c):
            for i in range(h_o):
                hs = i * stride
                he = hs + kernel
                for j in range(w_o):
                    ws = j * stride
                    we = ws + kernel
                    out[ch, i, j] = (
                        float(np.sum(input_tensor[ch, hs:he, ws:we])) / denom
                    )
        return out

    @staticmethod
    def _eval_conv3d(
        input_tensor: np.ndarray,
        filter_tensor: np.ndarray,
        stride: int,
        padding: str,
    ) -> np.ndarray:
        # Input: [C_in, D, H, W]
        # Filter: [C_out, C_in, Kd, Kh, Kw]
        in_shape = input_tensor.shape
        f_shape = filter_tensor.shape
        if len(in_shape) != 4:
            raise ValueError(f"Conv3D expects input rank 4 [C,D,H,W], got {in_shape}")
        if len(f_shape) != 5:
            raise ValueError(
                f"Conv3D expects filter rank 5 [Cout,Cin,Kd,Kh,Kw], got {f_shape}"
            )

        c_in, d_in, h_in, w_in = in_shape
        c_out, _c_in_f, k_d, k_h, k_w = f_shape
        if padding == "valid":
            d_o = (d_in - k_d) // stride + 1
            h_o = (h_in - k_h) // stride + 1
            w_o = (w_in - k_w) // stride + 1
            pad = (0, 0, 0, 0, 0, 0)
        elif padding == "same":
            if stride == 1:
                d_o, h_o, w_o = d_in, h_in, w_in
                total_pd = max(0, (d_o - 1) * stride + k_d - d_in)
                total_ph = max(0, (h_o - 1) * stride + k_h - h_in)
                total_pw = max(0, (w_o - 1) * stride + k_w - w_in)
                pf = total_pd // 2
                pb = total_pd - pf
                pt = total_ph // 2
                pbot = total_ph - pt
                pl = total_pw // 2
                pr = total_pw - pl
                pad = (pf, pb, pt, pbot, pl, pr)
            else:
                # Match the Conv2D evaluator convention: symmetric k//2 when stride>1.
                pf = pb = k_d // 2
                pt = pbot = k_h // 2
                pl = pr = k_w // 2
                d_pad = d_in + 2 * pf
                h_pad = h_in + 2 * pt
                w_pad = w_in + 2 * pl
                d_o = (d_pad - k_d) // stride + 1
                h_o = (h_pad - k_h) // stride + 1
                w_o = (w_pad - k_w) // stride + 1
                pad = (pf, pb, pt, pbot, pl, pr)
        else:
            raise ValueError(f"Unsupported padding mode: {padding!r}")

        pf, pb, pt, pbot, pl, pr = pad
        if any(x != 0 for x in pad):
            padded = np.zeros((c_in, d_in + pf + pb, h_in + pt + pbot, w_in + pl + pr))
            padded[:, pf : pf + d_in, pt : pt + h_in, pl : pl + w_in] = input_tensor
            input_tensor = padded

        out = np.zeros((c_out, d_o, h_o, w_o))
        for in_c in range(c_in):
            for out_c in range(c_out):
                f_in_idx = min(in_c, f_shape[1] - 1)
                for od in range(d_o):
                    d_start = od * stride
                    d_end = d_start + k_d
                    for oh in range(h_o):
                        h_start = oh * stride
                        h_end = h_start + k_h
                        for ow in range(w_o):
                            w_start = ow * stride
                            w_end = w_start + k_w
                            patch = input_tensor[
                                in_c, d_start:d_end, h_start:h_end, w_start:w_end
                            ]
                            out[out_c, od, oh, ow] += float(
                                np.sum(patch * filter_tensor[out_c, f_in_idx])
                            )
        return out

    def _eval_poly(
        self, x: np.ndarray, func: Any, inputs: Dict[str, Any]
    ) -> np.ndarray:
        if callable(func):
            return np.asarray(func(x))
        if func == "identity":
            return x
        # POLY("relu_exact"), PolyCall name "relu", or ``("relu", lo, hi)`` from PolyCallArgs.
        if (
            func == "relu_exact"
            or func == "relu"
            or (isinstance(func, tuple) and len(func) >= 1 and func[0] == "relu")
        ):
            return np.maximum(x, 0.0)
        if func == "silu":
            # Legacy wire format without explicit bounds; use a default interval for ``poly``.
            return eval_silu_polycall(
                np.asarray(x, dtype=np.float64), -8.0, 8.0, inputs
            )
        if isinstance(func, tuple) and len(func) >= 3 and func[0] == "silu":
            _, lo, hi = func[:3]
            return eval_silu_polycall(
                np.asarray(x, dtype=np.float64), float(lo), float(hi), inputs
            )
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
            case "Mean":
                ax = term.cs[1]
                axes = (ax,) if isinstance(ax, int) else tuple(ax)
                return np.mean(env[term.cs[0]], axis=axes, keepdims=True)
            case "Product":
                return np.prod(env[term.cs[0]], axis=term.cs[1], keepdims=False)
            case "Cast":
                return np.asarray(env[term.cs[0]]).astype(np.dtype(term.cs[1]))
            case "MatMul":
                return env[term.cs[0]] @ env[term.cs[1]]
            case "Transpose":
                return env[term.cs[0]].T
            case "Conv":
                from .tensor_args import Conv2dArgs

                args = Conv2dArgs.from_term(term)
                return self._eval_conv2d(
                    env[args.input],
                    env[args.filter],
                    args.stride,
                    args.padding,
                    args.groups,
                )
            case "Conv3D":
                from .tensor_args import Conv3dArgs

                args = Conv3dArgs.from_term(term)
                return self._eval_conv3d(
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

                    pargs = PolyCallArgs.from_term(term)
                    func_spec = (pargs.name, pargs.lower_bound, pargs.upper_bound)
                else:
                    func = term.cs[1] if len(term.cs) > 1 else "identity"
                    func_spec = func
                return self._eval_poly(x, func_spec, inputs)
            case "Concat":
                return np.concatenate(
                    [env[t] for t in term.cs[0]], axis=int(term.cs[1])
                )
            case "Tile":
                return np.tile(env[term.cs[0]], tuple(int(x) for x in term.cs[1]))
            case "CumSum":
                x = np.asarray(env[term.cs[0]])
                axis = int(term.cs[1])
                exclusive = bool(term.cs[2]) if len(term.cs) > 2 else False
                reverse = bool(term.cs[3]) if len(term.cs) > 3 else False
                if reverse:
                    x = np.flip(x, axis=axis)
                y = np.cumsum(x, axis=axis)
                if exclusive:
                    y = y - x
                if reverse:
                    y = np.flip(y, axis=axis)
                return self._trim_to_declared_shape(y, term)
            case "AvgPool2D":
                return self._eval_avg_pool2d(
                    env[term.cs[0]],
                    int(term.cs[1]),
                    int(term.cs[2]),
                    term.cs[3],
                )
            case "HardSwish":
                x = np.asarray(env[term.cs[0]], dtype=np.float64)
                y = x * np.clip(x + 3.0, 0.0, 6.0) / 6.0
                return self._trim_to_declared_shape(y, term)
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
