// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_sequence.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``sequence`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

/**
 * Computes the output :cpp:class:`OptimSequence` of a
 * ``SequenceConstruct`` node and stores it in ``ctx``.
 *
 * ``SequenceConstruct`` (since opset 11 in the ``ai.onnx`` domain) takes
 * ``N >= 1`` tensor inputs that share the same element type and
 * produces a single tensor-sequence output of length ``N``. The element
 * dtype of the output sequence is the common dtype of the inputs; the
 * ONNX schema does not require the inputs to share a common shape, so
 * the output :cpp:class:`OptimSequence` records one
 * :cpp:class:`OptimShape` per input verbatim (see
 * :cpp:func:`OptimSequence::ElemShapes`).
 *
 * When called with zero inputs, the output sequence has length ``0``,
 * an unknown element dtype (:cpp:enumerator:`TensorType::kUndefined`)
 * and an empty per-element shapes vector.
 *
 * @param ctx   In/out context. Must already contain an
 *              :cpp:class:`OptimTensor` entry for every named input of
 *              ``node``; on return it also contains an
 *              :cpp:class:`OptimSequence` entry for ``node.output(0)``.
 * @param node  The ``SequenceConstruct`` ``NodeProto`` whose output
 *              should be described. ``node.op_type()`` must be
 *              ``"SequenceConstruct"`` and ``node`` must declare at
 *              least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"SequenceConstruct"``, if ``node`` has no output, or if the
 *         input tensors do not share a common element dtype.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeSequenceConstruct(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``ConcatFromSequence``
 * node and stores it in ``ctx``.
 *
 * ``ConcatFromSequence`` (since opset 11 in the ``ai.onnx`` domain) takes
 * a single tensor-sequence input and produces a single tensor output by
 * concatenating (when ``new_axis == 0``, the default) or stacking (when
 * ``new_axis == 1``) the input tensors along ``axis``. The element type
 * of the output is the element type of the input sequence.
 *
 * The per-element shapes of the input sequence must all share the same
 * rank ``r``. When ``new_axis == 0``, ``axis`` ranges in ``[-r, r - 1]``
 * and the output has rank ``r``; the ``axis`` dimension is the sum of
 * the per-element dimensions along ``axis`` (or symbolic when any
 * per-element dimension along ``axis`` is symbolic), and every other
 * dimension is merged across elements (concrete values win over
 * symbolic, mismatched concrete values throw). When ``new_axis == 1``,
 * ``axis`` ranges in ``[-r - 1, r]`` and the output has rank ``r + 1``;
 * the new dimension at ``axis`` is the sequence length and every other
 * dimension is merged across elements.
 *
 * When the per-element shapes of the input sequence are unknown
 * (:cpp:func:`OptimSequence::HasElemShapes` is ``false``), only the
 * element dtype is recorded on the output and the shape is left
 * empty.
 *
 * @param ctx   In/out context. Must already contain an
 *              :cpp:class:`OptimSequence` entry for ``node.input(0)``;
 *              on return it also contains an :cpp:class:`OptimTensor`
 *              entry for ``node.output(0)``.
 * @param node  The ``ConcatFromSequence`` ``NodeProto`` whose output
 *              should be described.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"ConcatFromSequence"``, if ``node`` has no input or no
 *         output, if ``new_axis`` is not ``0`` or ``1``, if ``axis`` is
 *         out of range, or if the per-element shapes of the input
 *         sequence have inconsistent ranks or conflicting concrete
 *         dimensions on a non-concat axis.
 * @throws std::out_of_range     if the named input sequence is missing
 *         from ``ctx``.
 */
void ComputeShapeConcatFromSequence(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``SequenceLength``
 * node and stores it in ``ctx``.
 *
 * ``SequenceLength`` takes one sequence input and produces one scalar
 * INT64 tensor output. The output shape is always empty (rank 0).
 *
 * @param ctx   In/out context. Must already contain an
 *              :cpp:class:`OptimSequence` entry for ``node.input(0)``;
 *              on return it also contains an :cpp:class:`OptimTensor`
 *              entry for ``node.output(0)``.
 * @param node  The ``SequenceLength`` ``NodeProto`` whose output should
 *              be described.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"SequenceLength"``, if ``node`` has no input, or if
 *         ``node`` has no output.
 * @throws std::out_of_range     if the named input sequence is missing
 *         from ``ctx``.
 */
void ComputeShapeSequenceLength(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimSequence` of a ``SequenceEmpty``
 * node and stores it in ``ctx``.
 *
 * ``SequenceEmpty`` (since opset 11 in the ``ai.onnx`` domain) takes no
 * inputs and produces an empty sequence whose element dtype is taken
 * from the optional ``dtype`` attribute (an INT-valued
 * ``onnx::TensorProto::DataType``). When ``dtype`` is absent the schema
 * default is ``FLOAT``. The output sequence length is always ``0`` and
 * the per-element shapes vector is empty.
 *
 * @param ctx   In/out context. On return it contains an
 *              :cpp:class:`OptimSequence` entry for ``node.output(0)``.
 * @param node  The ``SequenceEmpty`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"SequenceEmpty"`` and ``node`` must declare at least
 *              one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"SequenceEmpty"``, if ``node`` has no output, or if the
 *         ``dtype`` attribute is present but is not an INT.
 */
void ComputeShapeSequenceEmpty(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimSequence` of a
 * ``SequenceErase`` node and stores it in ``ctx``.
 *
 * ``SequenceErase`` (since opset 11 in the ``ai.onnx`` domain) takes a
 * sequence input and an optional scalar position input, and produces a
 * sequence output with one element removed. The element dtype of the
 * output sequence matches the input sequence. When the input sequence
 * has known per-element shapes, the output sequence records the shapes
 * of all elements except the one at the erased position. Because the
 * position is a runtime value the erased index is generally unknown at
 * shape-inference time, so the output per-element shapes are dropped
 * and only the element dtype is forwarded together with a symbolic
 * sequence length.
 *
 * @param ctx   In/out context. Must already contain an
 *              :cpp:class:`OptimSequence` entry for ``node.input(0)``;
 *              on return it also contains an
 *              :cpp:class:`OptimSequence` entry for ``node.output(0)``.
 * @param node  The ``SequenceErase`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"SequenceErase"`` and ``node`` must declare at least
 *              one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"SequenceErase"`` or if ``node`` has no output.
 * @throws std::out_of_range     if the named input sequence is missing
 *         from ``ctx``.
 */
void ComputeShapeSequenceErase(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``SequenceAt`` node and
 * stores it in ``ctx``.
 *
 * ``SequenceAt`` (since opset 11 in the ``ai.onnx`` domain) takes a sequence
 * input and a required scalar position input, and produces a tensor output
 * equal to the sequence element at the given position. The output element
 * dtype matches the input sequence element dtype. Because the position is a
 * runtime value, the output shape is generally unknown; the only exception is
 * when the input sequence records per-element shapes and all of them are
 * equal, in which case the shared shape is forwarded as the output shape.
 *
 * @param ctx   In/out context. Must already contain an
 *              :cpp:class:`OptimSequence` entry for ``node.input(0)``; on
 *              return it also contains an :cpp:class:`OptimTensor` entry for
 *              ``node.output(0)``.
 * @param node  The ``SequenceAt`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"SequenceAt"`` and
 *              ``node`` must declare at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"SequenceAt"``
 *         or if ``node`` has no output.
 * @throws std::out_of_range     if the named input sequence is missing from
 *         ``ctx``.
 */
void ComputeShapeSequenceAt(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimSequence` of a
 * ``SequenceInsert`` node and stores it in ``ctx``.
 *
 * ``SequenceInsert`` (since opset 11 in the ``ai.onnx`` domain) takes a
 * sequence input, a tensor to insert, and an optional scalar position
 * input. The output sequence element dtype matches the input sequence
 * element dtype (or the inserted tensor dtype when the input sequence
 * dtype is unknown) and the output length is the input length plus one.
 * Because the insertion position is generally a runtime value, per-element
 * output shapes are not inferred.
 *
 * @param ctx   In/out context. Must already contain an
 *              :cpp:class:`OptimSequence` entry for ``node.input(0)`` and
 *              an :cpp:class:`OptimTensor` entry for ``node.input(1)``;
 *              on return it also contains an
 *              :cpp:class:`OptimSequence` entry for ``node.output(0)``.
 * @param node  The ``SequenceInsert`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"SequenceInsert"`` and ``node`` must declare at least
 *              one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"SequenceInsert"``, if ``node`` has fewer than two inputs, if
 *         ``node`` has no output, or if the sequence/tensor element dtypes
 *         disagree when both are known.
 * @throws std::out_of_range     if any required named input is missing
 *         from ``ctx``.
 */
void ComputeShapeSequenceInsert(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimSequence`(s) of a ``SequenceMap`` node
 * and stores them in ``ctx``.
 *
 * ``SequenceMap`` (since opset 17 in the ``ai.onnx`` domain) takes a
 * required sequence input (``input_sequence``) and zero or more additional
 * tensor or sequence inputs, plus a required ``body`` graph attribute. The
 * body subgraph is applied to each element of ``input_sequence`` together
 * with the additional inputs and produces ``M >= 1`` output tensors per
 * iteration. ``SequenceMap`` then produces ``M`` output sequences, each of
 * length equal to the length of ``input_sequence``.
 *
 * Shape inference walks the body subgraph in a child context seeded with:
 *
 * * body input ``0`` (the per-iteration element of ``input_sequence``):
 *   a tensor descriptor whose dtype matches the input sequence element
 *   dtype and whose shape is the common per-element shape of the input
 *   sequence (or empty when per-element shapes are not recorded);
 * * body inputs ``1..K`` (the additional inputs): inherited verbatim from
 *   the matching outer-scope ``node`` inputs (either tensor or sequence).
 *
 * Each output sequence then records the body output dtype as its element
 * dtype; the sequence length is the input sequence length (concrete when
 * the input length is known, otherwise symbolic). Per-element shapes are
 * not recorded on the output sequence (the body may vary the per-element
 * shape across iterations and we forward only the dtype).
 *
 * @param ctx   In/out context. Must already contain an
 *              :cpp:class:`OptimSequence` entry for ``node.input(0)`` and
 *              the matching :cpp:class:`OptimTensor` /
 *              :cpp:class:`OptimSequence` entries for the remaining
 *              ``node`` inputs; on return it also contains one
 *              :cpp:class:`OptimSequence` entry per declared
 *              ``node.output``.
 * @param node  The ``SequenceMap`` ``NodeProto`` whose outputs should be
 *              described. ``node.op_type()`` must be ``"SequenceMap"``,
 *              ``node`` must declare at least one input and one output,
 *              and must carry a graph attribute named ``"body"`` whose
 *              outputs match ``node.output_size()``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"SequenceMap"``, if ``node`` has no input or no output, if the
 *         ``body`` graph attribute is missing or has the wrong arity, or
 *         if a body output is missing from the inferred body context.
 * @throws std::out_of_range     if a named input is missing from ``ctx``.
 */
void ComputeShapeSequenceMap(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimSequence` of a ``SplitToSequence``
 * node and stores it in ``ctx``.
 *
 * ``SplitToSequence`` (since opset 11 in the ``ai.onnx`` domain) takes a
 * tensor input and an optional ``split`` tensor input, plus the
 * attributes ``axis`` (default ``0``) and ``keepdims`` (default ``1``,
 * ignored when ``split`` is provided), and produces a single
 * tensor-sequence output. The output element dtype matches the input
 * tensor dtype.
 *
 * Per-element shapes are inferred when possible:
 *
 * * when ``split`` is omitted, the input axis dimension is split into
 *   chunks of size ``1``; with ``keepdims == 1`` each element keeps the
 *   input rank with axis dim ``1``, with ``keepdims == 0`` the axis is
 *   squeezed away;
 * * when ``split`` is a 1-D tensor whose value is known at shape
 *   inference time, each entry gives the corresponding element's size
 *   along ``axis``;
 * * when ``split`` is a scalar whose value is known, equal chunks of
 *   that size are produced (the last chunk possibly being smaller).
 *
 * When the axis dimension or the ``split`` value are unknown the
 * sequence length and per-element shapes are dropped and only the
 * element dtype is forwarded together with a symbolic length.
 *
 * @param ctx   In/out context. Must already contain an
 *              :cpp:class:`OptimTensor` entry for ``node.input(0)``
 *              and, when present, for ``node.input(1)``; on return it
 *              also contains an :cpp:class:`OptimSequence` entry for
 *              ``node.output(0)``.
 * @param node  The ``SplitToSequence`` ``NodeProto`` whose output
 *              should be described. ``node.op_type()`` must be
 *              ``"SplitToSequence"`` and ``node`` must declare at
 *              least one input and one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"SplitToSequence"``, if ``node`` has no input or no
 *         output, if ``axis`` is out of range when the input rank is
 *         known, or if a known ``split`` tensor disagrees with the
 *         input axis dimension.
 * @throws std::out_of_range     if a named input is missing from
 *         ``ctx``.
 */
void ComputeShapeSplitToSequence(ShapesContext &ctx, const NodeProto &node);

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
