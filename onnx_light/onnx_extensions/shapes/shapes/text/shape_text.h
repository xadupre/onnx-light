// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_core/symbolic/sym_map.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_text.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``text`` family.
 */

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes {

// The generic shape-inference engine (ShapesContext, dispatch table,
// domain constants, ...) lives in ``onnx_core`` so it never depends on
// any particular set of operator implementations; the symbolic value
// descriptors (SymDim, SymShape, SymTensor, ...) live in
// ``onnx_core::symbolic`` for the same reason. Bring them into
// ``onnx_shapes::shapes`` so shape functions can keep referring to them
// unqualified, exactly as before those types moved to ``onnx_core``.
using ::onnx_light::core::shapes::BroadcastDimOp;
using ::onnx_light::core::shapes::BroadcastShapes;
using ::onnx_light::core::shapes::CheckNodeOpAndOutput;
using ::onnx_light::core::shapes::ComputeShapeBinaryBroadcast;
using ::onnx_light::core::shapes::InferShapesModel;
using ::onnx_light::core::shapes::kOnnxDomain;
using ::onnx_light::core::shapes::kUnknownOpsetVersion;
using ::onnx_light::core::shapes::PropagateValueAsShapeArithmetic;
using ::onnx_light::core::shapes::ShapeEvent;
using ::onnx_light::core::shapes::ShapeEventAction;
using ::onnx_light::core::shapes::ShapeEventActionName;
using ::onnx_light::core::shapes::ShapesContext;

using ::onnx_light::core::symbolic::DataTypeToTensorType;
using ::onnx_light::core::symbolic::Device;
using ::onnx_light::core::symbolic::GPUIndex;
using ::onnx_light::core::symbolic::IsGPU;
using ::onnx_light::core::symbolic::IsIntegerTensorType;
using ::onnx_light::core::symbolic::kMaxGPUIndex;
using ::onnx_light::core::symbolic::kMaxOptimRank;
using ::onnx_light::core::symbolic::kOptimValueAsShapeMaxElements;
using ::onnx_light::core::symbolic::kValueInfoDeviceMetadataKey;
using ::onnx_light::core::symbolic::kValueInfoMaxMetadataKey;
using ::onnx_light::core::symbolic::kValueInfoMinMetadataKey;
using ::onnx_light::core::symbolic::ShapeFromTensorProtoDims;
using ::onnx_light::core::symbolic::SymCmpResult;
using ::onnx_light::core::symbolic::SymDim;
using ::onnx_light::core::symbolic::SymMap;
using ::onnx_light::core::symbolic::SymSequence;
using ::onnx_light::core::symbolic::SymShape;
using ::onnx_light::core::symbolic::SymTensor;
using ::onnx_light::core::symbolic::TensorType;
using ::onnx_light::core::symbolic::TensorTypeToDataType;

namespace text {

/**
 * Computes the output :cpp:class:`SymTensor` of a ``StringConcat`` node
 * and stores it in ``ctx``.
 *
 * ``StringConcat`` concatenates two string tensors element-wise with
 * numpy-style multidirectional broadcasting (since opset 20 in the
 * ``ai.onnx`` domain). The output dtype is always
 * :cpp:enumerator:`TensorType::kString` and the output shape is the
 * broadcast of the two input shapes.
 *
 * @param ctx   In/out context. Must already contain entries for both
 *              ``a`` and ``b``; on return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``StringConcat`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"StringConcat"`` and ``node`` must declare at least
 *              one output.
 * @param a     Name of the first input value to read from ``ctx``.
 * @param b     Name of the second input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"StringConcat"``, if ``node`` has no output, or if the two
 *         input shapes are not broadcast-compatible.
 * @throws std::out_of_range     if either ``a`` or ``b`` is missing
 *         from ``ctx``.
 */
void ComputeShapeStringConcat(ShapesContext &ctx, const NodeProto &node, const char *a,
                              const char *b);

/**
 * Computes the output :cpp:class:`SymTensor` descriptors of a
 * ``StringSplit`` node and stores them in ``ctx``.
 *
 * ``StringSplit`` preserves the input rank for its ``Z`` output
 * (substring counts, ``tensor(int64)``) and appends one symbolic final
 * dimension to the input rank for its ``Y`` output (padded substrings,
 * ``tensor(string)``).
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``a``; on return it also contains entries for
 *              ``node.output(0)`` and ``node.output(1)``.
 * @param node  The ``StringSplit`` ``NodeProto`` whose outputs should
 *              be described. ``node.op_type()`` must be
 *              ``"StringSplit"`` and ``node`` must declare at least two
 *              outputs.
 * @param a     Name of the input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"StringSplit"``, if ``node`` declares fewer than two
 *         outputs, or if either output name is empty.
 * @throws std::out_of_range     if ``a`` is missing from ``ctx``.
 */
void ComputeShapeStringSplit(ShapesContext &ctx, const NodeProto &node, const char *a);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``StringNormalizer``
 * node and stores it in ``ctx``.
 *
 * ``StringNormalizer`` only accepts ``[C]``- or ``[1, C]``-shaped
 * ``tensor(string)`` inputs (since opset 10 in the ``ai.onnx`` domain).
 * The output dtype is always :cpp:enumerator:`TensorType::kString` and
 * has the same rank as the input. The last dimension of the output is
 * symbolic (``"StringNormalizer(<input>)"``) because it depends on how
 * many entries the ``stopwords`` attribute drops at runtime.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``a``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``StringNormalizer`` ``NodeProto`` whose output
 *              should be described. ``node.op_type()`` must be
 *              ``"StringNormalizer"`` and ``node`` must declare at
 *              least one output.
 * @param a     Name of the input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"StringNormalizer"``, if ``node`` has no output, or if
 *         the input shape has an unsupported rank.
 * @throws std::out_of_range     if ``a`` is missing from ``ctx``.
 */
void ComputeShapeStringNormalizer(ShapesContext &ctx, const NodeProto &node, const char *a);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``RegexFullMatch``
 * node and stores it in ``ctx``.
 *
 * ``RegexFullMatch`` (since opset 20 in the ``ai.onnx`` domain) performs an
 * element-wise full-match regex test on a ``tensor(string)`` input and
 * produces a ``tensor(bool)`` output of the same shape.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``a``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``RegexFullMatch`` ``NodeProto`` whose output
 *              should be described. ``node.op_type()`` must be
 *              ``"RegexFullMatch"`` and ``node`` must declare at
 *              least one output.
 * @param a     Name of the input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"RegexFullMatch"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``a`` is missing from ``ctx``.
 */
void ComputeShapeRegexFullMatch(ShapesContext &ctx, const NodeProto &node, const char *a);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``TfIdfVectorizer``
 * node and stores it in ``ctx``.
 *
 * ``TfIdfVectorizer`` (since opset 9 in the ``ai.onnx`` domain) extracts
 * n-grams from a ``[C]``- or ``[N, C]``-shaped integer / string input and
 * produces a ``tensor(float)`` whose last dimension is
 * ``max(ngram_indexes) + 1``. The output preserves the input batch
 * dimension when the input has rank 2.
 *
 * @param ctx   In/out context. Must already contain an entry for ``a``;
 *              on return it also contains an entry for ``node.output(0)``.
 * @param node  The ``TfIdfVectorizer`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"TfIdfVectorizer"`` and ``node`` must declare at least
 *              one output. The ``ngram_indexes`` attribute must be a
 *              non-empty list of non-negative ``int64``s.
 * @param a     Name of the input value to read from ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"TfIdfVectorizer"``, if ``node`` has no output, if
 *         ``ngram_indexes`` is missing/invalid, or if the input shape
 *         has an unsupported rank.
 * @throws std::out_of_range     if ``a`` is missing from ``ctx``.
 */
void ComputeShapeTfIdfVectorizer(ShapesContext &ctx, const NodeProto &node, const char *a);

} // namespace text
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes
