# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""NumPy overrides for selected operators in the native reference evaluator."""

from __future__ import annotations

import numpy as np

from ._evaluator import ReferenceEvaluator


def _attributes(node):
    """Returns the attributes of a node indexed by name."""
    return {str(attr.name): attr for attr in node.attribute}


def _gemm(node, a, b, c=None):
    """Computes Gemm using NumPy."""
    attrs = _attributes(node)
    if attrs.get("transA") and attrs["transA"].i:
        a = a.T
    if attrs.get("transB") and attrs["transB"].i:
        b = b.T
    alpha = float(attrs["alpha"].f) if "alpha" in attrs else 1.0
    beta = float(attrs["beta"].f) if "beta" in attrs else 1.0
    y = alpha * (a @ b)
    return y if c is None else y + beta * c


def _matmul(node, a, b):
    """Computes MatMul using NumPy."""
    return a @ b


def _conv(node, x, w, b=None):
    """Computes N-dimensional grouped convolution using NumPy."""
    attrs = _attributes(node)
    spatial = x.ndim - 2
    strides = tuple(attrs["strides"].ints) if "strides" in attrs else (1,) * spatial
    dilations = tuple(attrs["dilations"].ints) if "dilations" in attrs else (1,) * spatial
    group = int(attrs["group"].i) if "group" in attrs else 1
    kernel = w.shape[2:]
    effective = tuple((size - 1) * dilation + 1 for size, dilation in zip(kernel, dilations))
    pads = list(attrs["pads"].ints) if "pads" in attrs else [0] * (2 * spatial)
    auto_pad = attrs["auto_pad"].s if "auto_pad" in attrs else b"NOTSET"
    if isinstance(auto_pad, str):
        auto_pad = auto_pad.encode()
    if auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
        total = [
            max(0, (int(np.ceil(size / stride)) - 1) * stride + extent - size)
            for size, stride, extent in zip(x.shape[2:], strides, effective)
        ]
        pads = [v // 2 if auto_pad == b"SAME_UPPER" else (v + 1) // 2 for v in total]
        pads += [v - p for v, p in zip(total, pads)]
    elif auto_pad == b"VALID":
        pads = [0] * (2 * spatial)
    x = np.pad(x, [(0, 0), (0, 0), *zip(pads[:spatial], pads[spatial:])])
    windows = np.lib.stride_tricks.sliding_window_view(x, effective, axis=tuple(range(2, x.ndim)))
    windows = windows[
        (
            slice(None),
            slice(None),
            *(slice(None, None, stride) for stride in strides),
            *(slice(None, None, dilation) for dilation in dilations),
        )
    ]
    channels_per_group = x.shape[1] // group
    outputs_per_group = w.shape[0] // group
    outputs = []
    for g in range(group):
        part = windows[:, g * channels_per_group : (g + 1) * channels_per_group]
        weights = w[g * outputs_per_group : (g + 1) * outputs_per_group]
        result = np.tensordot(
            part,
            weights,
            axes=([1, *range(2 + spatial, 2 + 2 * spatial)], list(range(1, spatial + 2))),
        )
        outputs.append(np.moveaxis(result, -1, 1))
    y = np.concatenate(outputs, axis=1)
    return y if b is None else y + b.reshape((1, -1) + (1,) * spatial)


def _attention(node, q, k, v, mask=None, past_key=None, past_value=None, nonpad_kv_seqlen=None):
    """Computes scaled dot-product Attention using NumPy."""
    attrs = _attributes(node)
    output_dtype = q.dtype

    def get_int(name, default):
        return int(attrs[name].i) if name in attrs else default

    def get_float(name, default):
        return float(attrs[name].f) if name in attrs else default

    rank3 = q.ndim == 3
    if rank3:

        def promote(t, heads):
            return t.reshape(t.shape[0], t.shape[1], heads, -1).transpose(0, 2, 1, 3)

        q = promote(q, get_int("q_num_heads", 0))
        k = promote(k, get_int("kv_num_heads", 0))
        v = promote(v, get_int("kv_num_heads", 0))
    past = 0 if past_key is None else past_key.shape[2]
    if past_key is not None:
        k = np.concatenate((past_key, k), axis=2)
        v = np.concatenate((past_value, v), axis=2)
    if q.shape[1] != k.shape[1]:
        repeats = q.shape[1] // k.shape[1]
        k_scores = np.repeat(k, repeats, axis=1)
        v_scores = np.repeat(v, repeats, axis=1)
    else:
        k_scores, v_scores = k, v
    compute_dtype = np.float32 if output_dtype.itemsize < 4 else output_dtype
    scores = (q.astype(compute_dtype) @ k_scores.astype(compute_dtype).swapaxes(-1, -2)) * (
        get_float("scale", q.shape[-1] ** -0.5)
    )
    mode = get_int("qk_matmul_output_mode", 0)
    qk = scores.copy() if mode == 0 else None
    softcap = get_float("softcap", 0.0)
    if softcap > 0:
        scores = softcap * np.tanh(scores / softcap)
    if mode == 1:
        qk = scores.copy()
    offset = (
        nonpad_kv_seqlen[:, None, None, None] - q.shape[2]
        if nonpad_kv_seqlen is not None and past_key is None
        else past
    )
    q_positions = np.arange(q.shape[2])[:, None] + offset
    k_positions = np.arange(k.shape[2])[None, :]
    blocked = np.zeros((q.shape[2], k.shape[2]), dtype=bool)
    if get_int("is_causal", 0):
        blocked = blocked | (k_positions > q_positions)
    left = get_int("left_window_size", -1)
    right = get_int("right_window_size", -1)
    if left >= 0:
        blocked = blocked | (k_positions < q_positions - left)
    if right >= 0:
        blocked = blocked | (k_positions > q_positions + right)
    if nonpad_kv_seqlen is not None:
        blocked = blocked | (
            np.arange(k.shape[2])[None, None, None, :] >= nonpad_kv_seqlen[:, None, None, None]
        )
    scores = np.where(blocked, -np.inf, scores)
    if mask is not None:
        if mask.shape[-1] < k.shape[2]:
            mask = np.pad(
                mask,
                [(0, 0)] * (mask.ndim - 1) + [(0, k.shape[2] - mask.shape[-1])],
                constant_values=False if mask.dtype == np.bool_ else -np.inf,
            )
        scores = np.where(mask, scores, -np.inf) if mask.dtype == np.bool_ else scores + mask
    if mode == 2:
        qk = scores.copy()
    maximum = np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores - np.where(np.isfinite(maximum), maximum, 0))
    weights = np.where(np.isfinite(scores), weights, 0)
    denominator = weights.sum(axis=-1, keepdims=True)
    weights = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator != 0)
    if mode == 3:
        qk = weights.copy()
    y = (weights @ v_scores.astype(compute_dtype)).astype(output_dtype)
    if rank3:
        y = y.transpose(0, 2, 1, 3).reshape(y.shape[0], y.shape[2], -1)
    outputs = (y, k, v, None if qk is None else qk.astype(output_dtype))
    return outputs[: len(node.output)] if len(node.output) > 1 else y


class MixedReferenceEvaluator(ReferenceEvaluator):
    """Evaluates selected ONNX operators with NumPy and the rest with native kernels."""

    def __init__(self, proto, **kwargs):
        """Initializes the evaluator and registers NumPy implementations."""
        super().__init__(proto, **kwargs)
        for op_type, kernel in (
            ("Gemm", _gemm),
            ("MatMul", _matmul),
            ("Conv", _conv),
            ("Attention", _attention),
        ):
            self.register_custom_kernel("", op_type, kernel)
