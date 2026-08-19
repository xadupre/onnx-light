# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests rotary, causal-mask, and attention graph patterns."""

from __future__ import annotations

import unittest

import numpy as np

import onnx_light.onnx as onnxl
from onnx_light.ext_test_case import ExtTestCase, import_or_skip
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh

from onnx_light.onnx_core import optimization

ReferenceEvaluator = import_or_skip("onnx_light.onnx.reference", "ReferenceEvaluator")

BOOL = onnxl.TensorProto.BOOL
FLOAT = onnxl.TensorProto.FLOAT
FLOAT16 = onnxl.TensorProto.FLOAT16
INT64 = onnxl.TensorProto.INT64


def _initializer(name: str, value, dtype=None):
    """Creates an initializer from a NumPy-compatible value."""
    return onh.from_array(np.asarray(value, dtype=dtype), name=name)


def _value_info(name: str, dtype: int, shape):
    """Creates tensor value information."""
    return oh.make_tensor_value_info(name, dtype, shape)


def _make_model(nodes, inputs, outputs, initializers=(), *, opset: int = 18):
    """Creates a minimal model."""
    graph = oh.make_graph(nodes, "rotary_attention", inputs, outputs, list(initializers))
    return oh.make_model(
        graph, opset_imports=[oh.make_opsetid("", opset)], ir_version=11 if opset >= 24 else 10
    )


def _optimize(model, pattern_names, *, schema_lookup="default"):
    """Optimizes a model with only the requested patterns."""
    builder = (
        optimization.GraphBuilder(model)
        if schema_lookup == "default"
        else optimization.GraphBuilder(model, schema_lookup=schema_lookup)
    )
    graph = optimization.GraphGraph(
        builder, optimization.standard_patterns(pattern_names), use_global_patterns=False
    )
    rewrites = graph.optimize()
    optimized = builder.to_onnx("model")
    _assert_topological(optimized.graph)
    selected = set(pattern_names)
    return optimized, [rewrite for rewrite in rewrites if rewrite.pattern_name in selected]


def _assert_topological(graph) -> None:
    """Checks that every node input is available before use."""
    available = {value.name for value in graph.input}
    available.update(initializer.name for initializer in graph.initializer)
    for node in graph.node:
        missing = {name for name in node.input if name and name not in available}
        if missing:
            raise AssertionError(f"{node.op_type} reads unavailable inputs {sorted(missing)}")
        available.update(name for name in node.output if name)
    missing_outputs = {value.name for value in graph.output if value.name not in available}
    if missing_outputs:
        raise AssertionError(f"unavailable graph outputs {sorted(missing_outputs)}")


def _run(model, feeds):
    """Runs a model with the onnx-light reference evaluator."""
    return ReferenceEvaluator(model).run(None, feeds)


def _assert_equivalent(test, model, optimized, feeds, *, atol=1e-6) -> None:
    """Checks numerical equivalence for every graph output."""
    expected = _run(model, feeds)
    got = _run(optimized, feeds)
    test.assertEqual(len(expected), len(got))
    for expected_value, got_value in zip(expected, got):
        np.testing.assert_allclose(expected_value, got_value, atol=atol, rtol=atol)


def _range(shape, dtype=np.float32, *, bias=0.0):
    """Creates deterministic floating-point input data."""
    size = int(np.prod(shape))
    values = (np.arange(size, dtype=np.float32) + 1) / max(size, 1) + bias
    return values.reshape(shape).astype(dtype)


def _attribute(node, name: str):
    """Returns an integer or floating-point node attribute."""
    for attribute in node.attribute:
        if attribute.name == name:
            if attribute.type == onnxl.AttributeProto.INT:
                return attribute.i
            if attribute.type == onnxl.AttributeProto.FLOAT:
                return attribute.f
    raise AssertionError(f"attribute {name!r} not found on {node.op_type}")


def _make_rotary_concat_slice(*, negate_first: bool, gap: bool = False, nonzero: bool = False):
    """Builds the Slice/Concat rotary cancellation topology."""
    left_end = 3 if gap else 4
    right_start = 4
    left_width = left_end
    right_width = 8 - right_start
    left_zero_width = right_width
    right_zero_width = left_width
    zero_value = _initializer("zero_value", [1], np.float32) if nonzero else None
    constant_attributes = {"value": zero_value} if zero_value is not None else {}
    nodes = [
        oh.make_node("Slice", ["X", "start0", "end0", "axis"], ["x1"]),
        oh.make_node("Slice", ["X", "start1", "end1", "axis"], ["x2"]),
    ]
    if negate_first:
        nodes.extend(
            [
                oh.make_node("Neg", ["x1"], ["negative"]),
                oh.make_node(
                    "ConstantOfShape", ["left_zero_shape"], ["left_zero"], **constant_attributes
                ),
                oh.make_node(
                    "ConstantOfShape", ["right_zero_shape"], ["right_zero"], **constant_attributes
                ),
                oh.make_node("Concat", ["negative", "left_zero"], ["left"], axis=-1),
                oh.make_node("Concat", ["right_zero", "x2"], ["right"], axis=3),
            ]
        )
    else:
        nodes.extend(
            [
                oh.make_node("Neg", ["x2"], ["negative"]),
                oh.make_node(
                    "ConstantOfShape", ["left_zero_shape"], ["left_zero"], **constant_attributes
                ),
                oh.make_node(
                    "ConstantOfShape", ["right_zero_shape"], ["right_zero"], **constant_attributes
                ),
                oh.make_node("Concat", ["left_zero", "x1"], ["left"], axis=3),
                oh.make_node("Concat", ["negative", "right_zero"], ["right"], axis=-1),
            ]
        )
    nodes.append(oh.make_node("Add", ["left", "right"], ["Y"], name="rotary_add"))
    return _make_model(
        nodes,
        [_value_info("X", FLOAT, [1, 2, 3, 8])],
        [_value_info("Y", FLOAT, [1, 2, 3, 8])],
        [
            _initializer("start0", [0], np.int64),
            _initializer("end0", [left_end], np.int64),
            _initializer("start1", [right_start], np.int64),
            _initializer("end1", [8], np.int64),
            _initializer("axis", [3], np.int64),
            _initializer("left_zero_shape", [1, 2, 3, left_zero_width], np.int64),
            _initializer("right_zero_shape", [1, 2, 3, right_zero_width], np.int64),
        ],
    )


def _make_rotary_concat_split(*, zero_width: int = 8):
    """Builds the Split/Concat rotary cancellation topology."""
    nodes = [
        oh.make_node("Split", ["X", "split"], ["x1", "x2"], axis=1),
        oh.make_node("Neg", ["x1"], ["negative"]),
        oh.make_node("ConstantOfShape", ["zero_shape"], ["zero"]),
        oh.make_node("Concat", ["negative", "zero"], ["left"], axis=1),
        oh.make_node("Concat", ["zero", "x2"], ["right"], axis=1),
        oh.make_node("Add", ["left", "right"], ["Y"]),
    ]
    output_width = 16 if zero_width == 8 else 8 + zero_width
    return _make_model(
        nodes,
        [_value_info("X", FLOAT, [3, 16])],
        [_value_info("Y", FLOAT, [3, output_width])],
        [
            _initializer("split", [8, 8], np.int64),
            _initializer("zero_shape", [3, zero_width], np.int64),
        ],
    )


def _make_rotary_transpose_scatter():
    """Builds the Transpose/ScatterND rotary cancellation topology."""
    nodes = [
        oh.make_node("Split", ["X", "split"], ["x1", "x2"], axis=1),
        oh.make_node("Neg", ["x1"], ["negative"]),
        oh.make_node("ConstantOfShape", ["zero_shape"], ["zero1"]),
        oh.make_node("ConstantOfShape", ["zero_shape"], ["zero2"]),
        oh.make_node("Transpose", ["zero1"], ["zero1_t"], perm=[1, 0]),
        oh.make_node("Transpose", ["zero2"], ["zero2_t"], perm=[1, 0]),
        oh.make_node("Transpose", ["negative"], ["negative_t"], perm=[1, 0]),
        oh.make_node("Transpose", ["x2"], ["x2_t"], perm=[1, 0]),
        oh.make_node("ScatterND", ["zero1_t", "indices0", "negative_t"], ["scatter1"]),
        oh.make_node("ScatterND", ["zero2_t", "indices1", "x2_t"], ["scatter2"]),
        oh.make_node("Transpose", ["scatter1"], ["left"], perm=[1, 0]),
        oh.make_node("Transpose", ["scatter2"], ["right"], perm=[1, 0]),
        oh.make_node("Add", ["left", "right"], ["Y"]),
    ]
    return _make_model(
        nodes,
        [_value_info("X", FLOAT, [2, 4])],
        [_value_info("Y", FLOAT, [2, 4])],
        [
            _initializer("split", [2, 2], np.int64),
            _initializer("zero_shape", [2, 4], np.int64),
            _initializer("indices0", [[0], [1]], np.int64),
            _initializer("indices1", [[2], [3]], np.int64),
        ],
    )


def _make_half_rotary(dtype: int):
    """Builds a half-rotary decomposition."""
    nodes = [
        oh.make_node("Split", ["X"], ["x1", "x2"], axis=-1, num_outputs=2),
        oh.make_node("Neg", ["x2"], ["negative"]),
        oh.make_node("Concat", ["negative", "x1"], ["rotated"], axis=-1),
        oh.make_node("Mul", ["rotated", "sin"], ["rotated_sin"]),
        oh.make_node("Mul", ["X", "cos"], ["x_cos"]),
        oh.make_node("Add", ["rotated_sin", "x_cos"], ["Y"]),
    ]
    return _make_model(
        nodes,
        [
            _value_info("X", dtype, [1, 2, 3, 8]),
            _value_info("cos", dtype, [1, 1, 3, 8]),
            _value_info("sin", dtype, [1, 1, 3, 8]),
        ],
        [_value_info("Y", dtype, [1, 2, 3, 8])],
        opset=18,
    )


def _make_full_rotary(dtype: int, *, partial: bool):
    """Builds a full or partial rotary decomposition."""
    half_width = 2 if partial else 4
    nodes = [
        oh.make_node("Concat", ["cos_half", "cos_half"], ["cos"], axis=-1),
        oh.make_node("Concat", ["sin_half", "sin_half"], ["sin"], axis=-1),
    ]
    rotary_input = "X"
    if partial:
        nodes.append(oh.make_node("Split", ["X", "outer_split"], ["rotary", "tail"], axis=-1))
        rotary_input = "rotary"
    nodes.extend(
        [
            oh.make_node("Split", [rotary_input], ["x1", "x2"], axis=-1, num_outputs=2),
            oh.make_node("Neg", ["x2"], ["negative"]),
            oh.make_node("Concat", ["negative", "x1"], ["rotated"], axis=-1),
            oh.make_node("Mul", ["rotated", "sin"], ["rotated_sin"]),
            oh.make_node("Mul", [rotary_input, "cos"], ["x_cos"]),
            oh.make_node("Add", ["rotated_sin", "x_cos"], ["rotary_y"]),
        ]
    )
    output = "rotary_y"
    if partial:
        nodes.append(oh.make_node("Concat", ["rotary_y", "tail"], ["Y"], axis=-1))
        output = "Y"
    initializers = [_initializer("outer_split", [4, 4], np.int64)] if partial else []
    return _make_model(
        nodes,
        [
            _value_info("X", dtype, [1, 2, 3, 8]),
            _value_info("cos_half", dtype, [1, 1, 3, half_width]),
            _value_info("sin_half", dtype, [1, 1, 3, half_width]),
        ],
        [_value_info(output, dtype, [1, 2, 3, 8])],
        initializers,
        opset=23,
    )


def _make_causal_mask(*, shifted: bool, expose_range: bool = False):
    """Builds a causal-mask decomposition from input dimensions."""
    nodes = [
        oh.make_node("Shape", ["X"], ["dim1"], start=0, end=1),
        oh.make_node("Shape", ["X"], ["dim2"], start=1, end=2),
        oh.make_node("Squeeze", ["dim1"], ["s1"]),
        oh.make_node("Squeeze", ["dim2"], ["s2"]),
        oh.make_node("Range", ["zero", "s2", "one"], ["range1"]),
        oh.make_node("Range", ["s1", "s2", "one"], ["range2"]),
        oh.make_node("Unsqueeze", ["range1", "axes1"], ["u1"]),
        oh.make_node("Unsqueeze", ["range2", "axes2"], ["u2"]),
    ]
    if shifted:
        nodes.extend(
            [
                oh.make_node("Sub", ["u2", "shift"], ["shifted"]),
                oh.make_node("Greater", ["u1", "shifted"], ["mask"]),
            ]
        )
    else:
        nodes.append(oh.make_node("LessOrEqual", ["u1", "u2"], ["mask"]))
    outputs = [_value_info("mask", BOOL, [1, 1, 5, 2])]
    if expose_range:
        outputs.append(_value_info("range2", INT64, [2]))
    return _make_model(
        nodes,
        [_value_info("X", FLOAT, [3, 5])],
        outputs,
        [
            _initializer("zero", 0, np.int64),
            _initializer("one", 1, np.int64),
            _initializer("axes1", [0, 1, 2], np.int64),
            _initializer("axes2", [0, 1, 3], np.int64),
            _initializer("shift", [3], np.int64),
        ],
    )


def _make_causal_mask_mul_add():
    """Builds the causal mask Mul/Add decomposition."""
    nodes = [
        oh.make_node("Shape", ["X"], ["batch"], start=0, end=1),
        oh.make_node("Shape", ["X"], ["dim1"], start=1, end=2),
        oh.make_node("Shape", ["X"], ["dim2"], start=2, end=3),
        oh.make_node("Squeeze", ["dim1"], ["s1"]),
        oh.make_node("Squeeze", ["dim2"], ["s2"]),
        oh.make_node("Range", ["zero", "s1", "one"], ["range1"]),
        oh.make_node("Range", ["zero", "s2", "one"], ["range2"]),
        oh.make_node("Unsqueeze", ["range1", "axes1"], ["u1"]),
        oh.make_node("Unsqueeze", ["range2", "axes2"], ["u2"]),
        oh.make_node("Mul", ["u2", "batch"], ["scaled"]),
        oh.make_node("Add", ["u1", "scaled"], ["mask"]),
    ]
    return _make_model(
        nodes,
        [_value_info("X", FLOAT, [2, 3, 4])],
        [_value_info("mask", INT64, [1, 3, 4, 1])],
        [
            _initializer("zero", 0, np.int64),
            _initializer("one", 1, np.int64),
            _initializer("axes1", [0, 1, 2], np.int64),
            _initializer("axes2", [1, 2, 3], np.int64),
        ],
    )


def _make_cos_sin_cache(*, output_dtype: int, position_ids: bool, consumers: bool = False):
    """Builds a cosine/sine cache decomposition."""
    inputs = [_value_info("weights", FLOAT, [1, 1, 8])]
    nodes = []
    initializers = [_initializer("reshape_shape", [0, -1, 1], np.int64)]
    if position_ids:
        inputs.insert(0, _value_info("position_ids", INT64, [3]))
        nodes.append(oh.make_node("Unsqueeze", ["position_ids", "axes"], ["positions_u"]))
        initializers.append(_initializer("axes", [1], np.int64))
    else:
        inputs.insert(0, _value_info("dim1", INT64, [1]))
        inputs.insert(1, _value_info("dim2", INT64, [1]))
        nodes.extend(
            [
                oh.make_node("Squeeze", ["dim1"], ["s1"]),
                oh.make_node("Squeeze", ["dim2"], ["s2"]),
                oh.make_node("Range", ["s1", "s2", "one"], ["positions"]),
                oh.make_node("Unsqueeze", ["positions", "axes"], ["positions_u"]),
            ]
        )
        initializers.extend(
            [_initializer("one", 1, np.int64), _initializer("axes", [0, 1], np.int64)]
        )
    nodes.extend(
        [
            oh.make_node("Cast", ["positions_u"], ["positions_f"], to=FLOAT),
            oh.make_node("Reshape", ["positions_f", "reshape_shape"], ["positions_r"]),
            oh.make_node("Mul", ["weights", "positions_r"], ["weighted"]),
            oh.make_node("Cos", ["weighted"], ["cos_raw"]),
            oh.make_node("Sin", ["weighted"], ["sin_raw"]),
        ]
    )
    cos_output = "cos_raw"
    sin_output = "sin_raw"
    if output_dtype != FLOAT:
        nodes.extend(
            [
                oh.make_node("Cast", ["cos_raw"], ["cos_cache"], to=output_dtype),
                oh.make_node("Cast", ["sin_raw"], ["sin_cache"], to=output_dtype),
            ]
        )
        cos_output = "cos_cache"
        sin_output = "sin_cache"
    if consumers:
        nodes.extend(
            [
                oh.make_node("Exp", [cos_output], ["cos_output"]),
                oh.make_node("Exp", [sin_output], ["sin_output"]),
            ]
        )
        cos_output = "cos_output"
        sin_output = "sin_output"
    shape = [3, 1, 8] if position_ids else [1, 3, 8]
    return _make_model(
        nodes,
        inputs,
        [
            _value_info(cos_output, output_dtype, shape),
            _value_info(sin_output, output_dtype, shape),
        ],
        initializers,
    )


def _make_attention(dtype: int, *, rank3: bool, switched_mask: bool = False):
    """Builds a three- or four-dimensional attention decomposition."""
    numpy_dtype = np.float16 if dtype == FLOAT16 else np.float32
    nodes = []
    inputs = []
    if rank3:
        inputs.extend(
            [
                _value_info("query", dtype, [1, 3, 8]),
                _value_info("keys", dtype, [1, 5, 8]),
                _value_info("values", dtype, [1, 5, 8]),
            ]
        )
        nodes.extend(
            [
                oh.make_node("Mul", ["query", "scale"], ["query_scaled_3d"]),
                oh.make_node("Mul", ["keys", "scale"], ["keys_scaled_3d"]),
                oh.make_node("Reshape", ["query_scaled_3d", "query_shape"], ["query_r"]),
                oh.make_node("Reshape", ["keys_scaled_3d", "key_value_shape"], ["keys_r"]),
                oh.make_node("Reshape", ["values", "key_value_shape"], ["values_r"]),
                oh.make_node("Transpose", ["query_r"], ["query_t"], perm=[0, 2, 1, 3]),
                oh.make_node("Transpose", ["keys_r"], ["keys_t"], perm=[0, 2, 3, 1]),
                oh.make_node("Transpose", ["values_r"], ["values_t"], perm=[0, 2, 1, 3]),
                oh.make_node("MatMul", ["query_t", "keys_t"], ["scores"]),
            ]
        )
        values = "values_t"
        initializers = [
            _initializer("query_shape", [0, 0, 2, 4], np.int64),
            _initializer("key_value_shape", [0, 0, 2, 4], np.int64),
        ]
    else:
        inputs.extend(
            [
                _value_info("query", dtype, [1, 2, 3, 4]),
                _value_info("keys", dtype, [1, 2, 5, 4]),
                _value_info("values", dtype, [1, 2, 5, 4]),
            ]
        )
        query = "query"
        keys = "keys"
        values = "values"
        initializers = []
    inputs.append(_value_info("mask", BOOL, [1, 1, 3, 5]))
    if not rank3:
        nodes.extend(
            [
                oh.make_node("Mul", [query, "scale"], ["query_scaled"]),
                oh.make_node("Mul", [keys, "scale"], ["keys_scaled"]),
            ]
        )
        nodes.append(
            oh.make_node("Transpose", ["keys_scaled"], ["keys_scaled_t"], perm=[0, 1, 3, 2])
        )
        nodes.append(oh.make_node("MatMul", ["query_scaled", "keys_scaled_t"], ["scores"]))
    if switched_mask:
        nodes.append(oh.make_node("Where", ["mask", "minfty", "scores"], ["masked"]))
    else:
        nodes.extend(
            [
                oh.make_node("Where", ["mask", "zero", "minfty"], ["bias"]),
                oh.make_node("Add", ["scores", "bias"], ["masked"]),
            ]
        )
    nodes.extend(
        [
            oh.make_node("Softmax", ["masked"], ["probabilities"], axis=-1),
            oh.make_node("IsNaN", ["probabilities"], ["nan"]),
            oh.make_node("Where", ["nan", "zero", "probabilities"], ["filtered"]),
            oh.make_node("MatMul", ["filtered", values], ["Y"]),
        ]
    )
    initializers.extend(
        [
            _initializer("zero", [0], numpy_dtype),
            _initializer("minfty", [-np.inf], numpy_dtype),
            _initializer("scale", [0.5], numpy_dtype),
        ]
    )
    return _make_model(nodes, inputs, [_value_info("Y", dtype, [1, 2, 3, 4])], initializers)


def _make_gqa_attention(
    *, opset: int, repeat_after_scale: bool, with_cache: bool, dtype: int = FLOAT
):
    """Builds a GQA decomposition with repeat-interleave branches."""
    numpy_dtype = np.float16 if dtype == FLOAT16 else np.float32
    inputs = [
        _value_info("query", dtype, [1, 4, 3, 2]),
        _value_info("key", dtype, [1, 2, 2, 2]),
        _value_info("value", dtype, [1, 2, 2, 2]),
        _value_info("mask", BOOL, [1, 1, 3, 3 if with_cache else 2]),
    ]
    nodes = []
    key_source = "key"
    value_source = "value"
    outputs = [_value_info("Y", dtype, [1, 4, 3, 2])]
    if with_cache:
        inputs.extend(
            [
                _value_info("past_key", dtype, [1, 2, 1, 2]),
                _value_info("past_value", dtype, [1, 2, 1, 2]),
            ]
        )
        nodes.extend(
            [
                oh.make_node("Concat", ["past_key", "key"], ["present_key"], axis=2),
                oh.make_node("Concat", ["past_value", "value"], ["present_value"], axis=2),
            ]
        )
        key_source = "present_key"
        value_source = "present_value"
        outputs.extend(
            [
                _value_info("present_key", dtype, [1, 2, 3, 2]),
                _value_info("present_value", dtype, [1, 2, 3, 2]),
            ]
        )
    nodes.append(oh.make_node("Mul", ["query", "scale"], ["query_scaled"]))
    nodes.append(oh.make_node("Unsqueeze", [key_source, "axis2"], ["key_u"]))
    if repeat_after_scale:
        nodes.append(oh.make_node("Mul", ["key_u", "scale"], ["key_u_scaled"]))
        nodes.append(oh.make_node("Expand", ["key_u_scaled", "expand_shape"], ["key_e"]))
        nodes.append(oh.make_node("Reshape", ["key_e", "reshape_shape"], ["key_r"]))
        key_scaled = "key_r"
    else:
        nodes.append(oh.make_node("Expand", ["key_u", "expand_shape"], ["key_e"]))
        nodes.append(oh.make_node("Reshape", ["key_e", "reshape_shape"], ["key_r"]))
        nodes.append(oh.make_node("Mul", ["key_r", "scale"], ["key_scaled"]))
        key_scaled = "key_scaled"
    nodes.extend(
        [
            oh.make_node("Unsqueeze", [value_source, "axis2"], ["value_u"]),
            oh.make_node("Expand", ["value_u", "expand_shape"], ["value_e"]),
            oh.make_node("Reshape", ["value_e", "reshape_shape"], ["value_r"]),
            oh.make_node("Transpose", [key_scaled], ["key_t"], perm=[0, 1, 3, 2]),
            oh.make_node("MatMul", ["query_scaled", "key_t"], ["scores"]),
            oh.make_node("Where", ["mask", "minfty", "scores"], ["masked"]),
            oh.make_node("Softmax", ["masked"], ["probabilities"], axis=-1),
            oh.make_node("IsNaN", ["probabilities"], ["nan"]),
            oh.make_node("Where", ["nan", "zero", "probabilities"], ["filtered"]),
            oh.make_node("MatMul", ["filtered", "value_r"], ["Y"]),
        ]
    )
    return _make_model(
        nodes,
        inputs,
        outputs,
        [
            _initializer("axis2", [2], np.int64),
            _initializer("expand_shape", [1, 1, 2, 1, 1], np.int64),
            _initializer("reshape_shape", [0, 4, -1, 2], np.int64),
            _initializer("scale", [0.5], numpy_dtype),
            _initializer("zero", [0], numpy_dtype),
            _initializer("minfty", [-np.inf], numpy_dtype),
        ],
        opset=opset,
    )


def _make_attention_gqa(
    *, dtype: int, mask_dtype: int, reshape: bool, opset: int = 24, active_optional: bool = False
):
    """Builds ONNX Attention preceded by cache and GQA expansion."""
    nodes = [
        oh.make_node("Concat", ["past_key", "key"], ["present_key"], axis=2),
        oh.make_node("Concat", ["past_value", "value"], ["present_value"], axis=2),
        oh.make_node("Unsqueeze", ["present_key", "axis2"], ["key_u"]),
        oh.make_node("Expand", ["key_u", "expand_shape"], ["key_e"]),
    ]
    if reshape:
        nodes.append(oh.make_node("Reshape", ["key_e", "reshape_shape"], ["key_r"]))
    else:
        nodes.append(oh.make_node("Squeeze", ["key_e", "axis1"], ["key_r"]))
    nodes.extend(
        [
            oh.make_node("Unsqueeze", ["present_value", "axis2"], ["value_u"]),
            oh.make_node("Expand", ["value_u", "expand_shape"], ["value_e"]),
        ]
    )
    if reshape:
        nodes.append(oh.make_node("Reshape", ["value_e", "reshape_shape"], ["value_r"]))
    else:
        nodes.append(oh.make_node("Squeeze", ["value_e", "axis1"], ["value_r"]))
    attention_inputs = ["query", "key_r", "value_r", "mask"]
    if active_optional:
        attention_inputs.append("active_optional")
    nodes.append(oh.make_node("Attention", attention_inputs, ["Y"], scale=0.11))
    key_value_heads = 2 if reshape else 1
    query_heads = 4 if reshape else 2
    model = _make_model(
        nodes,
        [
            _value_info("query", dtype, [1, query_heads, 3, 4]),
            _value_info("key", dtype, [1, key_value_heads, 2, 4]),
            _value_info("value", dtype, [1, key_value_heads, 2, 4]),
            _value_info("mask", mask_dtype, [1, 1, 3, 3]),
            _value_info("past_key", dtype, [1, key_value_heads, 1, 4]),
            _value_info("past_value", dtype, [1, key_value_heads, 1, 4]),
        ]
        + (
            [_value_info("active_optional", dtype, [1, query_heads, 1, 4])]
            if active_optional
            else []
        ),
        [
            _value_info("Y", dtype, [1, query_heads, 3, 4]),
            _value_info("present_key", dtype, [1, key_value_heads, 3, 4]),
            _value_info("present_value", dtype, [1, key_value_heads, 3, 4]),
        ],
        [
            _initializer("axis1", [1], np.int64),
            _initializer("axis2", [2], np.int64),
            _initializer("expand_shape", [1, 1, 2, 1, 1], np.int64),
            _initializer("reshape_shape", [0, query_heads, -1, 4], np.int64),
        ],
        opset=opset,
    )
    model.graph.value_info.extend(
        [
            _value_info("key_r", dtype, [1, query_heads, 3, 4]),
            _value_info("value_r", dtype, [1, query_heads, 3, 4]),
        ]
    )
    return model


class TestRotaryAttentionPatterns(ExtTestCase):
    """Tests migrated xoptim rotary and attention scenarios."""

    def test_rotary_concat_part_slice_first_half(self):
        model = _make_rotary_concat_slice(negate_first=True)
        optimized, rewrites = _optimize(model, ["RotaryConcatPart"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node], ["Slice", "Slice", "Neg", "Concat"]
        )
        _assert_equivalent(self, model, optimized, {"X": _range((1, 2, 3, 8))})

    def test_rotary_concat_part_slice_second_half(self):
        model = _make_rotary_concat_slice(negate_first=False)
        optimized, rewrites = _optimize(model, ["RotaryConcatPart"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node], ["Slice", "Slice", "Neg", "Concat"]
        )
        _assert_equivalent(self, model, optimized, {"X": _range((1, 2, 3, 8))})

    def test_rotary_concat_part_split(self):
        model = _make_rotary_concat_split()
        optimized, rewrites = _optimize(model, ["RotaryConcatPart"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node], ["Split", "Neg", "Concat"]
        )
        _assert_equivalent(self, model, optimized, {"X": _range((3, 16))})

    def test_rotary_concat_part_transpose_scatter(self):
        model = _make_rotary_transpose_scatter()
        optimized, rewrites = _optimize(model, ["RotaryConcatPart"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node], ["Split", "Neg", "Concat"]
        )
        _assert_equivalent(self, model, optimized, {"X": _range((2, 4))})

    def test_rotary_concat_part_wrong_zero_shape_no_match(self):
        model = _make_rotary_concat_split(zero_width=16)
        optimized, rewrites = _optimize(model, ["RotaryConcatPart"])
        self.assertEqual(len(rewrites), 0)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node],
            ["Split", "Neg", "ConstantOfShape", "Concat", "Concat", "Add"],
        )
        _assert_equivalent(self, model, optimized, {"X": _range((3, 16))})

    def test_rotary_concat_part_gap_no_match(self):
        model = _make_rotary_concat_slice(negate_first=True, gap=True)
        optimized, rewrites = _optimize(model, ["RotaryConcatPart"])
        self.assertEqual(len(rewrites), 0)
        _assert_equivalent(self, model, optimized, {"X": _range((1, 2, 3, 8))})

    def test_rotary_concat_part_nonzero_padding_no_match(self):
        model = _make_rotary_concat_slice(negate_first=True, nonzero=True)
        optimized, rewrites = _optimize(model, ["RotaryConcatPart"])
        self.assertEqual(len(rewrites), 0)
        _assert_equivalent(self, model, optimized, {"X": _range((1, 2, 3, 8))})

    def test_function_half_rotary_embedding_float32(self):
        model = _make_half_rotary(FLOAT)
        feeds = {
            "X": _range((1, 2, 3, 8)),
            "cos": _range((1, 1, 3, 8), bias=0.1),
            "sin": _range((1, 1, 3, 8), bias=0.2),
        }
        optimized, rewrites = _optimize(model, ["FunctionHalfRotaryEmbedding"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["HalfRotaryEmbedding"])
        self.assertEqual(
            [(function.domain, function.name) for function in optimized.functions],
            [("intermediate", "HalfRotaryEmbedding")],
        )
        _assert_equivalent(self, model, optimized, feeds)

    def test_function_half_rotary_embedding_float16(self):
        model = _make_half_rotary(FLOAT16)
        feeds = {
            "X": _range((1, 2, 3, 8), np.float16),
            "cos": _range((1, 1, 3, 8), np.float16, bias=0.1),
            "sin": _range((1, 1, 3, 8), np.float16, bias=0.2),
        }
        optimized, rewrites = _optimize(model, ["FunctionHalfRotaryEmbedding"])
        self.assertEqual(len(rewrites), 1)
        _assert_equivalent(self, model, optimized, feeds, atol=2e-3)

    def test_function_half_rotary_embedding_wrong_axis_no_match(self):
        model = _make_model(
            [oh.make_node("Split", ["X"], ["x1", "x2"], axis=2, num_outputs=2)],
            [_value_info("X", FLOAT, [1, 2, 4, 8])],
            [_value_info("x1", FLOAT, [1, 2, 2, 8]), _value_info("x2", FLOAT, [1, 2, 2, 8])],
        )
        optimized, rewrites = _optimize(model, ["FunctionHalfRotaryEmbedding"])
        self.assertEqual(len(rewrites), 0)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Split"])

    def test_rotary_embedding_full_no_split_float32(self):
        model = _make_full_rotary(FLOAT, partial=False)
        feeds = {
            "X": _range((1, 2, 3, 8)),
            "cos_half": _range((1, 1, 3, 4), bias=0.1),
            "sin_half": _range((1, 1, 3, 4), bias=0.2),
        }
        optimized, rewrites = _optimize(model, ["FunctionHalfRotaryEmbedding", "RotaryEmbedding"])
        self.assertEqual(
            [rewrite.pattern_name for rewrite in rewrites],
            ["FunctionHalfRotaryEmbedding", "RotaryEmbedding"],
        )
        rotary = next(node for node in optimized.graph.node if node.op_type == "RotaryEmbedding")
        self.assertEqual(_attribute(rotary, "num_heads"), 2)
        self.assertFalse(
            any(attribute.name == "rotary_embedding_dim" for attribute in rotary.attribute)
        )
        _assert_equivalent(self, model, optimized, feeds, atol=1e-5)

    def test_rotary_embedding_partial_split_float16(self):
        model = _make_full_rotary(FLOAT16, partial=True)
        feeds = {
            "X": _range((1, 2, 3, 8), np.float16),
            "cos_half": _range((1, 1, 3, 2), np.float16, bias=0.1),
            "sin_half": _range((1, 1, 3, 2), np.float16, bias=0.2),
        }
        optimized, rewrites = _optimize(model, ["FunctionHalfRotaryEmbedding", "RotaryEmbedding"])
        self.assertEqual(len(rewrites), 2)
        rotary = next(node for node in optimized.graph.node if node.op_type == "RotaryEmbedding")
        self.assertEqual(_attribute(rotary, "rotary_embedding_dim"), 4)
        _assert_equivalent(self, model, optimized, feeds, atol=3e-3)

    def test_rotary_embedding_opset22_no_match(self):
        model = _make_full_rotary(FLOAT, partial=False)
        model.opset_import[0].version = 22
        optimized, rewrites = _optimize(model, ["FunctionHalfRotaryEmbedding", "RotaryEmbedding"])
        self.assertEqual(len(rewrites), 1)
        self.assertIn("HalfRotaryEmbedding", [node.op_type for node in optimized.graph.node])
        self.assertNotIn("RotaryEmbedding", [node.op_type for node in optimized.graph.node])

    def test_function_causal_mask(self):
        model = _make_causal_mask(shifted=False)
        optimized, rewrites = _optimize(model, ["FunctionCausalMask"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node], ["Shape", "Shape", "CausalMask"]
        )
        self.assertEqual(optimized.functions[0].name, "CausalMask")
        _assert_equivalent(self, model, optimized, {"X": _range((3, 5))})

    def test_function_shifted_causal_mask(self):
        model = _make_causal_mask(shifted=True)
        optimized, rewrites = _optimize(model, ["FunctionCausalMask"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node],
            ["Shape", "Shape", "ShiftedCausalMask"],
        )
        _assert_equivalent(self, model, optimized, {"X": _range((3, 5))})

    def test_function_causal_mask_mul_add(self):
        model = _make_causal_mask_mul_add()
        optimized, rewrites = _optimize(model, ["FunctionCausalMaskMulAdd"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node],
            ["Shape", "Shape", "Shape", "CausalMaskMulAdd"],
        )
        _assert_equivalent(self, model, optimized, {"X": _range((2, 3, 4))})

    def test_function_causal_mask_bug(self):
        model = _make_causal_mask(shifted=False, expose_range=True)
        optimized, rewrites = _optimize(model, ["FunctionCausalMask"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node],
            ["Shape", "Shape", "Squeeze", "Squeeze", "Range", "CausalMask"],
        )
        self.assertEqual([output.name for output in optimized.graph.output], ["mask", "range2"])
        _assert_equivalent(self, model, optimized, {"X": _range((3, 5))})

    def test_function_cos_sin_cache_float16(self):
        model = _make_cos_sin_cache(output_dtype=FLOAT16, position_ids=False)
        feeds = {
            "dim1": np.array([3], dtype=np.int64),
            "dim2": np.array([6], dtype=np.int64),
            "weights": _range((1, 1, 8)),
        }
        optimized, rewrites = _optimize(model, ["FunctionCosSinCache"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node], ["CosSinCacheWithRange_to10"]
        )
        self.assertEqual(len(optimized.graph.node[0].output), 2)
        _assert_equivalent(self, model, optimized, feeds, atol=2e-3)

    def test_function_cos_sin_cache_float16_position_ids(self):
        model = _make_cos_sin_cache(output_dtype=FLOAT16, position_ids=True)
        feeds = {
            "position_ids": np.array([3, 5, 6], dtype=np.int64),
            "weights": _range((1, 1, 8)),
        }
        optimized, rewrites = _optimize(model, ["FunctionCosSinCache"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["CosSinCache_to10_p1"])
        _assert_equivalent(self, model, optimized, feeds, atol=2e-3)

    def test_function_cos_sin_cache_float32(self):
        model = _make_cos_sin_cache(output_dtype=FLOAT, position_ids=False)
        feeds = {
            "dim1": np.array([3], dtype=np.int64),
            "dim2": np.array([6], dtype=np.int64),
            "weights": _range((1, 1, 8)),
        }
        optimized, rewrites = _optimize(model, ["FunctionCosSinCache"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node], ["CosSinCacheWithRange"]
        )
        _assert_equivalent(self, model, optimized, feeds)

    def test_function_cos_sin_cache_consumers_no_match(self):
        model = _make_cos_sin_cache(output_dtype=FLOAT, position_ids=False, consumers=True)
        feeds = {
            "dim1": np.array([3], dtype=np.int64),
            "dim2": np.array([6], dtype=np.int64),
            "weights": _range((1, 1, 8)),
        }
        optimized, rewrites = _optimize(model, ["FunctionCosSinCache"])
        self.assertEqual(len(rewrites), 0)
        self.assertEqual([node.op_type for node in optimized.graph.node][-2:], ["Exp", "Exp"])
        _assert_equivalent(self, model, optimized, feeds)

    def test_function_attention_3d(self):
        model = _make_attention(FLOAT, rank3=True)
        feeds = {
            "query": _range((1, 3, 8)),
            "keys": _range((1, 5, 8), bias=0.1),
            "values": _range((1, 5, 8), bias=0.2),
            "mask": np.array([[[[True, True, False, True, True]] * 3]], dtype=np.bool_),
        }
        optimized, rewrites = _optimize(model, ["FunctionAttention"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node],
            [
                "Reshape",
                "Transpose",
                "Reshape",
                "Reshape",
                "Transpose",
                "Transpose",
                "LocalAttention_to1",
            ],
        )
        _assert_equivalent(self, model, optimized, feeds, atol=1e-5)

    def test_function_attention_4d_float32(self):
        model = _make_attention(FLOAT, rank3=False)
        feeds = {
            "query": _range((1, 2, 3, 4)),
            "keys": _range((1, 2, 5, 4), bias=0.1),
            "values": _range((1, 2, 5, 4), bias=0.2),
            "mask": np.array([[[[True, True, False, True, True]] * 3]], dtype=np.bool_),
        }
        optimized, rewrites = _optimize(model, ["FunctionAttention"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["LocalAttention_to1"])
        _assert_equivalent(self, model, optimized, feeds)

    def test_function_attention_4d_float16_switched_mask(self):
        model = _make_attention(FLOAT16, rank3=False, switched_mask=True)
        optimized, rewrites = _optimize(model, ["FunctionAttention"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node], ["LocalAttentionSW_to10"]
        )
        self.assertEqual(optimized.functions[0].name, "LocalAttentionSW_to10")

    def test_function_attention_positive_infinity_no_match(self):
        model = _make_attention(FLOAT, rank3=False)
        infinity = next(
            initializer for initializer in model.graph.initializer if initializer.name == "minfty"
        )
        infinity.raw_data = np.array([np.inf], dtype=np.float32).tobytes()
        optimized, rewrites = _optimize(model, ["FunctionAttention"])
        self.assertEqual(len(rewrites), 0)
        self.assertIn("Softmax", [node.op_type for node in optimized.graph.node])

    def test_function_attention_gqa_detected_during_attention(self):
        model = _make_gqa_attention(opset=22, repeat_after_scale=True, with_cache=False)
        feeds = {
            "query": _range((1, 4, 3, 2)),
            "key": _range((1, 2, 2, 2), bias=0.1),
            "value": _range((1, 2, 2, 2), bias=0.2),
            "mask": np.zeros((1, 1, 3, 2), dtype=np.bool_),
        }
        optimized, rewrites = _optimize(model, ["FunctionAttention"])
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(
            [node.op_type for node in optimized.graph.node], ["LocalAttentionGQASW_to1"]
        )
        _assert_equivalent(self, model, optimized, feeds, atol=1e-5)

    def test_function_attention_gqa_second_stage(self):
        model = _make_gqa_attention(opset=22, repeat_after_scale=False, with_cache=False)
        feeds = {
            "query": _range((1, 4, 3, 2)),
            "key": _range((1, 2, 2, 2), bias=0.1),
            "value": _range((1, 2, 2, 2), bias=0.2),
            "mask": np.zeros((1, 1, 3, 2), dtype=np.bool_),
        }
        optimized, rewrites = _optimize(model, ["FunctionAttention", "FunctionAttentionGQA"])
        self.assertEqual(
            [rewrite.pattern_name for rewrite in rewrites],
            ["FunctionAttention", "FunctionAttentionGQA"],
        )
        self.assertEqual(
            [node.op_type for node in optimized.graph.node], ["LocalAttentionGQASW_to1"]
        )
        _assert_equivalent(self, model, optimized, feeds, atol=1e-5)

    def test_attention_gqa_default_opset22_stays_local(self):
        model = _make_gqa_attention(opset=22, repeat_after_scale=False, with_cache=True)
        optimized, rewrites = _optimize(
            model, ["FunctionAttention", "FunctionAttentionGQA", "AttentionGQA"]
        )
        self.assertEqual(len(rewrites), 2)
        self.assertIn("LocalAttentionGQASW_to1", [node.op_type for node in optimized.graph.node])
        self.assertNotIn("Attention", [node.op_type for node in optimized.graph.node])

    def test_attention_gqa_default_opset24_fuses_caches(self):
        model = _make_gqa_attention(opset=24, repeat_after_scale=False, with_cache=True)
        feeds = {
            "query": _range((1, 4, 3, 2)),
            "key": _range((1, 2, 2, 2), bias=0.1),
            "value": _range((1, 2, 2, 2), bias=0.2),
            "mask": np.zeros((1, 1, 3, 3), dtype=np.bool_),
            "past_key": _range((1, 2, 1, 2), bias=0.3),
            "past_value": _range((1, 2, 1, 2), bias=0.4),
        }
        optimized, rewrites = _optimize(
            model, ["FunctionAttention", "FunctionAttentionGQA", "AttentionGQA"]
        )
        self.assertEqual(len(rewrites), 3)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Not", "Attention"])
        self.assertEqual(
            list(optimized.graph.node[-1].output), ["Y", "present_key", "present_value"]
        )
        _assert_equivalent(self, model, optimized, feeds, atol=1e-5)

    def test_attention_gqa_bool_squeeze_float32(self):
        model = _make_attention_gqa(dtype=FLOAT, mask_dtype=BOOL, reshape=False)
        feeds = {
            "query": _range((1, 2, 3, 4)),
            "key": _range((1, 1, 2, 4), bias=0.1),
            "value": _range((1, 1, 2, 4), bias=0.2),
            "mask": np.array([[[[True, False, True]] * 3]], dtype=np.bool_),
            "past_key": _range((1, 1, 1, 4), bias=0.3),
            "past_value": _range((1, 1, 1, 4), bias=0.4),
        }
        optimized, rewrites = _optimize(model, ["AttentionGQA"], schema_lookup=None)
        self.assertEqual(len(rewrites), 1)
        self.assertEqual([node.op_type for node in optimized.graph.node], ["Attention"])
        self.assertEqual(len(optimized.graph.node[0].output), 3)
        _assert_equivalent(self, model, optimized, feeds, atol=1e-5)

    def test_attention_gqa_bool_reshape_float16(self):
        model = _make_attention_gqa(dtype=FLOAT16, mask_dtype=BOOL, reshape=True)
        feeds = {
            "query": _range((1, 4, 3, 4), np.float16),
            "key": _range((1, 2, 2, 4), np.float16, bias=0.1),
            "value": _range((1, 2, 2, 4), np.float16, bias=0.2),
            "mask": np.array([[[[True, False, True]] * 3]], dtype=np.bool_),
            "past_key": _range((1, 2, 1, 4), np.float16, bias=0.3),
            "past_value": _range((1, 2, 1, 4), np.float16, bias=0.4),
        }
        optimized, rewrites = _optimize(model, ["AttentionGQA"], schema_lookup=None)
        self.assertEqual(len(rewrites), 1)
        _assert_equivalent(self, model, optimized, feeds, atol=3e-3)

    def test_attention_gqa_float_mask(self):
        model = _make_attention_gqa(dtype=FLOAT, mask_dtype=FLOAT, reshape=False)
        feeds = {
            "query": _range((1, 2, 3, 4)),
            "key": _range((1, 1, 2, 4), bias=0.1),
            "value": _range((1, 1, 2, 4), bias=0.2),
            "mask": np.array([[[[0.0, -np.inf, 0.0]] * 3]], dtype=np.float32),
            "past_key": _range((1, 1, 1, 4), bias=0.3),
            "past_value": _range((1, 1, 1, 4), bias=0.4),
        }
        optimized, rewrites = _optimize(model, ["AttentionGQA"], schema_lookup=None)
        self.assertEqual(len(rewrites), 1)
        _assert_equivalent(self, model, optimized, feeds, atol=1e-5)

    def test_attention_gqa_old_opset_no_match(self):
        model = _make_attention_gqa(dtype=FLOAT, mask_dtype=BOOL, reshape=False, opset=22)
        optimized, rewrites = _optimize(model, ["AttentionGQA"], schema_lookup=None)
        self.assertEqual(len(rewrites), 0)
        self.assertEqual([node.op_type for node in optimized.graph.node][-1], "Attention")

    def test_attention_gqa_active_optional_input_no_match(self):
        model = _make_attention_gqa(
            dtype=FLOAT, mask_dtype=BOOL, reshape=False, active_optional=True
        )
        optimized, rewrites = _optimize(model, ["AttentionGQA"], schema_lookup=None)
        self.assertEqual(len(rewrites), 0)
        self.assertEqual(len(optimized.graph.node), 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
