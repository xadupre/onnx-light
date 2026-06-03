// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_tensor.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``tensor`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Concat`` node
 * and stores it in ``ctx``.
 *
 * ``Concat`` concatenates a variadic list of input tensors along the
 * axis specified by the ``axis`` attribute. All inputs must share the
 * same rank and the same dimension sizes on every axis other than the
 * concatenation axis. The output dtype always matches the dtype of
 * the first input (type constraint ``T``); the output shape is:
 *
 *   - on the concatenation axis: the sum of all input dimensions when
 *     every input dimension on that axis is a concrete integer;
 *     otherwise a fresh symbolic dimension;
 *   - on every other axis: the merged dimension between all inputs
 *     (concrete dimensions must match across inputs, otherwise an
 *     exception is thrown; a concrete value overrides a symbolic
 *     one).
 *
 * The ``axis`` attribute can be negative, in which case it is
 * interpreted as ``axis + rank``. When the attribute is missing the
 * default of ``1`` (the opset 1 default) is used.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              every name in ``node.input``. On return it also
 *              contains an entry for ``node.output(0)``.
 * @param node  The ``Concat`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Concat"``,
 *              ``node`` must declare at least one input and at least
 *              one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Concat"``, if ``node`` has no input or output, if the
 *         inputs have different ranks, if their non-concat dimensions
 *         disagree, if the resolved axis is out of range, or if the
 *         input dtypes differ.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeConcat(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Cast`` node and
 * stores it in ``ctx``.
 *
 * ``Cast`` produces an output whose shape is identical to the shape of
 * its single input and whose element type is given by the required
 * integer attribute ``to`` (a ``TensorProto::DataType`` value). The
 * other optional attributes (``saturate``, ``round_mode``) do not
 * affect the output shape or dtype and are therefore not inspected by
 * this function.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry
 *              for ``node.output(0)``.
 * @param node  The ``Cast`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Cast"``,
 *              ``node`` must declare at least one input and at least
 *              one output and must carry the required ``to``
 *              attribute.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Cast"``, if ``node`` has no input or output, if the
 *         ``to`` attribute is missing, or if its value does not map to
 *         a supported :cpp:enum:`TensorType`.
 * @throws std::out_of_range     if the input name is missing from
 *         ``ctx``.
 */
void ComputeShapeCast(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``BitCast`` node
 * (opset 26) and stores it in ``ctx``.
 *
 * ``BitCast`` reinterprets the bit pattern of its input as the data type
 * specified by the required ``to`` attribute. The output shape always
 * matches the input shape; the output dtype is taken from ``to``. The
 * target dtype must have the same per-element bit-width as the input dtype
 * (the upstream ONNX schema enforces this rule); if the widths differ this
 * function throws ``std::invalid_argument``.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``BitCast`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"BitCast"`` and
 *              ``node`` must declare at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"BitCast"``,
 *         if ``node`` has no output, if the ``to`` attribute is missing,
 *         if its value does not map to a supported (non-string)
 *         :cpp:enum:`TensorType`, or if the input and output element
 *         bit-widths differ.
 * @throws std::out_of_range     if the input name is missing from ``ctx``.
 */
void ComputeShapeBitCast(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``CastLike`` node and
 * stores it in ``ctx``.
 *
 * ``CastLike`` produces an output whose shape is identical to the shape of
 * its first input (``input``) and whose element type is taken from the
 * second input (``target_type``); the values of ``target_type`` are
 * ignored. The optional attributes (``saturate``, ``round_mode``) do not
 * affect the output shape or dtype and are therefore not inspected by this
 * function.
 *
 * @param ctx   In/out context. Must already contain entries for
 *              ``node.input(0)`` and ``node.input(1)``. On return it also
 *              contains an entry for ``node.output(0)``.
 * @param node  The ``CastLike`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"CastLike"``,
 *              ``node`` must declare two inputs and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"CastLike"``,
 *         if ``node`` has fewer than two inputs or no output, or if the
 *         dtype of ``target_type`` is :cpp:enum:`TensorType::kUndefined`.
 * @throws std::out_of_range     if any input name is missing from ``ctx``.
 */
void ComputeShapeCastLike(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Reshape`` node
 * and stores it in ``ctx``.
 *
 * ``Reshape`` takes a ``data`` tensor and a 1-D int64 ``shape`` tensor
 * whose values describe the desired output shape. The output dtype is
 * the dtype of ``data`` (type constraint ``T``). The output shape is
 * derived element-by-element from the target shape:
 *
 *   - a positive value is used verbatim;
 *   - ``0`` means "copy from the input ``data`` shape at the same
 *     index", unless the ``allowzero`` attribute is set to ``1`` (in
 *     which case ``0`` is honoured literally);
 *   - exactly one ``-1`` is allowed; the corresponding dimension is
 *     inferred so that the total number of elements is preserved
 *     (when ``data`` is fully known and the other dims are concrete);
 *   - symbolic target dims are forwarded as symbolic output dims.
 *
 * Shape values are read from the ``shape`` input's
 * :cpp:func:`OptimTensor::ValueAsShape` annotation (populated for
 * small constants, e.g. by :cpp:func:`ComputeShapeConstant`). When
 * that annotation is missing the output rank is taken from the static
 * shape of the ``shape`` input (its single dimension, when concrete)
 * and every output dim is left symbolic. When the rank itself is
 * unknown the output is left as a fully-symbolic rank-1 tensor.
 *
 * @param ctx   In/out context. Must contain entries for
 *              ``node.input(0)`` (``data``) and ``node.input(1)``
 *              (``shape``). On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Reshape`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Reshape"``,
 *              ``node`` must declare two inputs and at least one
 *              output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Reshape"``, if ``node`` has fewer than two inputs or no
 *         output, if the target shape contains more than one ``-1``,
 *         contains a value strictly less than ``-1``, if a ``0`` entry
 *         (with ``allowzero == 0``) refers to a position outside the
 *         input rank, or if a ``-1`` cannot be reconciled with the
 *         input's element count.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeReshape(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Slice`` node and stores
 * it in ``ctx``.
 *
 * ``Slice`` preserves input rank and dtype. When ``starts``/``ends`` (and
 * optional ``axes``/``steps``) values are known through
 * :cpp:func:`OptimTensor::ValueAsShape`, concrete output lengths are inferred
 * per sliced axis; otherwise sliced axes are left symbolic.
 */
void ComputeShapeSlice(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Expand`` node
 * and stores it in ``ctx``.
 *
 * ``Expand`` broadcasts its first input (``input``) to the shape
 * described by its second input (``shape``), following the ONNX
 * numpy-style broadcasting rules. The output dtype matches the dtype
 * of ``input`` (type constraint ``T``).
 *
 * Shape values are read from the ``shape`` input's
 * :cpp:func:`OptimTensor::ValueAsShape` annotation (populated for
 * small constants). When that annotation is present the output shape
 * is computed as ``BroadcastShapes(input.shape, target)``. When it is
 * absent the output rank is taken from the static shape of the
 * ``shape`` input (its single dimension, when concrete) and every
 * output dim is left symbolic.
 *
 * @param ctx   In/out context. Must already contain entries for
 *              ``node.input(0)`` (``input``) and ``node.input(1)``
 *              (``shape``). On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Expand`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Expand"``,
 *              ``node`` must declare two inputs and at least one
 *              output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Expand"``, if ``node`` has fewer than two inputs or no
 *         output, or if the two shapes are not broadcast-compatible.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeExpand(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Squeeze`` node and
 * stores it in ``ctx``.
 *
 * ``Squeeze`` removes dimensions of size 1 from the input shape. When the
 * optional ``axes`` input is provided and its values are known, only those
 * axes are removed (and each selected axis must be 1 when concrete). When
 * ``axes`` is omitted, every concrete unit dimension is removed.
 */
void ComputeShapeSqueeze(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Unsqueeze`` node and
 * stores it in ``ctx``.
 *
 * ``Unsqueeze`` inserts dimensions of size 1 at the indices given by the
 * required ``axes`` input.
 */
void ComputeShapeUnsqueeze(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Transpose`` node
 * and stores it in ``ctx``.
 *
 * ``Transpose`` permutes the axes of its input tensor according to the
 * optional ``perm`` attribute. When ``perm`` is absent, axes are reversed.
 * The output dtype matches the input dtype.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Transpose`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Transpose"``,
 *              ``node`` must declare one input and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Transpose"``,
 *         if ``node`` has no input or output, if ``perm`` length differs from
 *         input rank, if ``perm`` contains an out-of-range value, or if it
 *         contains duplicates.
 * @throws std::out_of_range     if the input name is missing from ``ctx``.
 */
/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Tile`` node and
 * stores it in ``ctx``.
 *
 * ``Tile`` constructs a tensor by repeating its first input (``input``)
 * a number of times along each axis given by the 1-D INT64 ``repeats``
 * tensor. The output has the same rank and dtype as ``input`` (type
 * constraint ``T``); its dimension ``i`` is ``input.shape[i] * repeats[i]``.
 *
 * Repeats values are read from the ``repeats`` input's
 * :cpp:func:`OptimTensor::ValueAsShape` annotation (populated for small
 * constants). When that annotation is present each output dim is computed
 * as ``input.shape[i] * repeats[i]`` (the multiplication is performed
 * symbolically when ``input.shape[i]`` is not a concrete integer; when
 * ``repeats[i]`` is symbolic the output dim is left symbolic). When the
 * annotation is absent the output rank is taken from the static rank of
 * ``input`` and every output dim is left symbolic.
 *
 * @param ctx   In/out context. Must already contain entries for
 *              ``node.input(0)`` (``input``) and ``node.input(1)``
 *              (``repeats``). On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Tile`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Tile"``, ``node``
 *              must declare two inputs and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Tile"``,
 *         if ``node`` has fewer than two inputs or no output, or if a
 *         known ``repeats`` input has a length different from the rank
 *         of ``input``.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeTile(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Upsample`` node and
 * stores it in ``ctx``. Supports Upsample opsets 1, 7, 9 and 10:
 *
 * - v1: the per-spatial-axis ``width_scale`` and ``height_scale`` FLOAT
 *   attributes give the scales of the two trailing axes of a 4-D NCHW
 *   input. Output dim ``k`` is ``floor(input_dim[k] * scale[k])``.
 * - v7: the ``scales`` FLOATS attribute carries one scale per input axis.
 * - v9/v10: the ``scales`` input tensor (1-D FLOAT) carries one scale per
 *   input axis. Because the data-propagation lattice only tracks integer
 *   shape values, the float scales cannot be recovered here in general;
 *   the output rank is preserved and every output dim is left symbolic.
 *
 * The output dtype always matches the input dtype (type constraint ``T``)
 * and the output rank equals the input rank.
 */
void ComputeShapeUpsample(ShapesContext &ctx, const NodeProto &node);

void ComputeShapeTranspose(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``DepthToSpace`` node and
 * stores it in ``ctx``.
 *
 * ``DepthToSpace`` requires a rank-4 input of shape ``(N, C, H, W)`` and a
 * required positive integer attribute ``blocksize``. The output dtype matches
 * the input dtype (type constraint ``T``) and the output shape is
 * ``(N, C/(blocksize*blocksize), H*blocksize, W*blocksize)`` (each axis is
 * computed symbolically when the corresponding input dim is symbolic).
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``DepthToSpace`` ``NodeProto`` whose output should be
 *              described.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"DepthToSpace"``, if ``node`` has no input or output, if the input
 *         rank is known and is not 4, if ``blocksize`` is missing or
 *         non-positive, or if the input channel dim is concrete and not
 *         divisible by ``blocksize * blocksize``.
 * @throws std::out_of_range     if the input name is missing from ``ctx``.
 */
void ComputeShapeDepthToSpace(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``SpaceToDepth`` node and
 * stores it in ``ctx``.
 *
 * ``SpaceToDepth`` requires a rank-4 input of shape ``(N, C, H, W)`` and a
 * required positive integer attribute ``blocksize``. The output dtype matches
 * the input dtype (type constraint ``T``) and the output shape is
 * ``(N, C*blocksize*blocksize, H/blocksize, W/blocksize)`` (each axis is
 * computed symbolically when the corresponding input dim is symbolic).
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``SpaceToDepth`` ``NodeProto`` whose output should be
 *              described.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"SpaceToDepth"``, if ``node`` has no input or output, if the input
 *         rank is known and is not 4, if ``blocksize`` is missing or
 *         non-positive, or if the input H or W dim is concrete and not
 *         divisible by ``blocksize``.
 * @throws std::out_of_range     if the input name is missing from ``ctx``.
 */
void ComputeShapeSpaceToDepth(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``AffineGrid`` node
 * and stores it in ``ctx``.
 *
 * ``AffineGrid`` produces a flow field of sampling coordinates from a
 * batch of affine matrices ``theta`` and a target ``size``. The output
 * dtype matches ``theta``'s dtype (type constraint ``T1``).
 *
 * The output shape is derived as follows:
 *
 *   - If ``size`` exposes a value via :cpp:func:`OptimTensor::ValueAsShape`
 *     (typically because it is a Constant), the spatial output dims
 *     ``(H, W)`` for 2-D / ``(D, H, W)`` for 3-D are taken verbatim from
 *     ``size`` and the rank of the output is fully known. ``size`` must
 *     have 4 (2-D) or 5 (3-D) entries.
 *   - Otherwise the rank of ``theta`` determines whether the output is
 *     2-D (``theta`` shape ``(N, 2, 3)``) or 3-D (``theta`` shape
 *     ``(N, 3, 4)``), and the spatial dims are left symbolic.
 *
 * The leading batch dim ``N`` is taken from ``theta[0]``; the final
 * inner dim is the constant ``2`` (2-D) or ``3`` (3-D).
 *
 * @param ctx   In/out context. Must already contain entries for
 *              ``theta`` and ``size``. On return it also contains an
 *              entry for ``node.output(0)``.
 * @param node  The ``AffineGrid`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"AffineGrid"``,
 *              ``node`` must declare two inputs and at least one
 *              output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"AffineGrid"``, if ``node`` has fewer than two inputs or no
 *         output, if ``theta`` is not rank 3, if its inner dims are
 *         neither ``(2, 3)`` nor ``(3, 4)``, or if a known ``size``
 *         input has a length other than 4 or 5.
 * @throws std::out_of_range     if any input name is missing from
 *         ``ctx``.
 */
void ComputeShapeAffineGrid(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``GridSample`` node and
 * stores it in ``ctx``.
 *
 * ``GridSample`` samples an input tensor ``X`` of rank ``r+2`` and shape
 * ``(N, C, D1, D2, ..., Dr)`` at the normalised locations given by a flow
 * field ``grid`` of rank ``r+2`` and shape
 * ``(N, D1_out, D2_out, ..., Dr_out, r)``, producing an output of rank
 * ``r+2`` and shape ``(N, C, D1_out, D2_out, ..., Dr_out)``. The output
 * dtype matches ``X``'s dtype (type constraint ``T1``).
 *
 * The output shape is derived as follows (each dim independently):
 *
 *   - ``output[0]``: merged dim between ``X[0]`` and ``grid[0]``.
 *   - ``output[1]``: ``X[1]`` (the channel dim).
 *   - ``output[2 .. r+1]``: the spatial dims taken from ``grid[1 .. r]``.
 *
 * @param ctx   In/out context. Must already contain entries for ``X`` and
 *              ``grid``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``GridSample`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"GridSample"``,
 *              ``node`` must declare two inputs and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"GridSample"``, if ``node`` has fewer than two inputs or no
 *         output, if ``X`` and ``grid`` have known ranks that disagree, or
 *         if either rank is below 3.
 * @throws std::out_of_range     if any input name is missing from ``ctx``.
 */
void ComputeShapeGridSample(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``NonZero`` node and
 * stores it in ``ctx``.
 *
 * ``NonZero`` returns the indices of the non-zero elements of its single input
 * tensor (in row-major order). The output is always an :cpp:enum:`TensorType::kInt64`
 * 2-D tensor of shape ``(rank, nnz)`` where ``rank`` is the rank of the input
 * and ``nnz`` is the number of non-zero elements. For scalar input
 * (``rank == 0``) the upstream spec dictates an output shape of ``(0, nnz)``,
 * which differs from NumPy's behaviour.
 *
 * Because the number of non-zero elements is a runtime value, the second
 * output dimension is left symbolic. The first output dimension is concrete
 * when the input rank is known and otherwise symbolic.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``NonZero`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"NonZero"``,
 *              ``node`` must declare one input and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"NonZero"``,
 *         or if ``node`` has no input or output.
 * @throws std::out_of_range     if the input name is missing from ``ctx``.
 */
void ComputeShapeNonZero(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Shape`` node and stores
 * it in ``ctx``.
 *
 * ``Shape`` returns a 1-D :cpp:enum:`TensorType::kInt64` tensor whose entries
 * are the dimensions of its input. Optional ``start`` and ``end`` attributes
 * (since opset 15) bound the slice ``input.shape[start:end]``: negative
 * values count from the back and out-of-range values are clamped to
 * ``[0, r]`` where ``r`` is the input rank. When ``start > end`` (after
 * normalisation) the output is empty.
 *
 * The output dimension is concrete when the input rank is known (which is
 * always the case when an :cpp:class:`OptimTensor` is available in
 * ``ctx``). The data buffer of the input is never inspected.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Shape`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"Shape"``; ``node`` must
 *              declare one input and at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Shape"``,
 *         or if ``node`` has no input or output.
 * @throws std::out_of_range     if the input name is missing from ``ctx``.
 */
void ComputeShapeShape(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Gather`` node and stores
 * it in ``ctx``.
 *
 * ``Gather`` indexes the ``data`` tensor along ``axis`` using the integer
 * ``indices`` tensor. The output has rank ``q + (r - 1)`` where
 * ``r = rank(data)`` and ``q = rank(indices)``; concretely the output shape is
 * ``data.shape[:axis] + indices.shape + data.shape[axis+1:]``. The output
 * dtype matches the dtype of ``data`` (type constraint ``T``).
 */
void ComputeShapeGather(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``GatherElements`` node
 * and stores it in ``ctx``.
 *
 * The output has the same shape as ``indices`` and the same dtype as ``data``.
 */
void ComputeShapeGatherElements(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``GatherND`` node and
 * stores it in ``ctx``.
 *
 * The output has rank ``q + r - indices_shape[-1] - 1 - b`` where ``b`` is
 * the ``batch_dims`` attribute (defaulting to ``0``); concretely the output
 * shape is ``indices.shape[:-1] + data.shape[b + indices.shape[-1]:]``.
 */
void ComputeShapeGatherND(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Trilu`` node and stores
 * it in ``ctx``.
 *
 * ``Trilu`` returns the upper (``upper`` attribute = 1, the default) or lower
 * (``upper`` = 0) triangular part of the input tensor; the optional ``k``
 * input shifts the diagonal. The output has the same dtype and the same shape
 * as ``node.input(0)``; the optional ``k`` input is not consulted for shape
 * inference (its value only affects element values, not the result shape).
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``node.input(0)``. On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Trilu`` ``NodeProto`` whose output should be described.
 *              ``node.op_type()`` must be ``"Trilu"``, ``node`` must declare
 *              at least one input and at least one output, and the rank of
 *              the input must be ``>= 2``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Trilu"``,
 *         if ``node`` has no input or output, or if the input rank is < 2.
 * @throws std::out_of_range     if the input name is missing from ``ctx``.
 */
void ComputeShapeTrilu(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``ReverseSequence`` node
 * and stores it in ``ctx``.
 *
 * ``ReverseSequence`` reverses the first ``sequence_lens[i]`` elements of
 * each slice along the time axis. The output has the same dtype and the same
 * shape as ``node.input(0)``; the ``sequence_lens`` input only affects
 * element values, not the result shape.
 *
 * @param ctx   In/out context. Must already contain entries for
 *              ``node.input(0)`` (input) and ``node.input(1)``
 *              (``sequence_lens``). On return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``ReverseSequence`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"ReverseSequence"``,
 *              ``node`` must declare two inputs and at least one output, and
 *              the rank of the first input must be ``>= 2`` while the rank of
 *              ``sequence_lens`` must be exactly 1.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"ReverseSequence"``, if ``node`` has fewer than two inputs or no
 *         output, or if the input ranks are invalid.
 * @throws std::out_of_range     if an input name is missing from ``ctx``.
 */
void ComputeShapeReverseSequence(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Compress`` node and
 * stores it in ``ctx``.
 *
 * When the ``axis`` attribute is present the output has the same rank and
 * dtype as ``node.input(0)`` but with the axis dimension replaced by a
 * symbolic dimension (the number of ``true`` entries in ``condition`` is a
 * runtime value). When ``axis`` is absent the input is conceptually flattened
 * and the output is a 1-D tensor of symbolic length.
 *
 * @param ctx   In/out context. Must already contain entries for
 *              ``node.input(0)`` (input) and ``node.input(1)`` (condition).
 *              On return it also contains an entry for ``node.output(0)``.
 * @param node  The ``Compress`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Compress"`` and
 *              ``node`` must declare at least two inputs and one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Compress"``
 *         or if the node has fewer than two inputs / no output.
 * @throws std::out_of_range     if an input name is missing from ``ctx``.
 */
void ComputeShapeCompress(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the per-output :cpp:class:`OptimTensor` of a ``Split`` node and
 * stores them in ``ctx``.
 *
 * ``Split`` divides ``input`` along ``axis`` into ``node.output_size()``
 * tensors. The split sizes are taken from (in order of priority):
 *
 *   - the ``split`` input (opset 13 and above) when present and known as an
 *     initializer value;
 *   - the ``split`` attribute (opset 1, 2 and 11) when present;
 *   - the ``num_outputs`` attribute (opset 18+) when present;
 *   - otherwise the input dimension is divided evenly into
 *     ``node.output_size()`` chunks (with the last chunk taking the
 *     remainder).
 *
 * Each output has the same rank, dtype, and dimensions as ``input``, except
 * along ``axis`` where the dimension equals the resolved split size when
 * known. When the split sizes are unknown (for example because the
 * ``split`` input is dynamic) the per-output ``axis`` dimension is set to a
 * fresh symbolic dimension.
 *
 * @param ctx   In/out context. Must already contain entries for every name
 *              in ``node.input``. On return it also contains entries for
 *              every named output.
 * @param node  The ``Split`` ``NodeProto`` whose outputs should be
 *              described. ``node.op_type()`` must be ``"Split"`` and
 *              ``node`` must declare at least one input and one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Split"``,
 *         if ``node`` has no input or no output, if the resolved axis is
 *         out of range, or if the resolved split sizes do not sum to the
 *         input dimension on ``axis``.
 * @throws std::out_of_range     if any input name is missing from ``ctx``.
 */
void ComputeShapeSplit(ShapesContext &ctx, const NodeProto &node);

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
