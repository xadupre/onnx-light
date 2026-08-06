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
 * @file shape_generator.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``generator`` family.
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

namespace generator {

/// Maximum element count of a ``Constant`` value tensor for which
/// :cpp:func:`ComputeShapeConstant` populates the output
/// :cpp:func:`SymTensor::ValueAsShape` annotation. Constants beyond
/// this threshold are not data-propagated (the output dtype and shape
/// are still inferred normally). Kept in sync with
/// :cpp:var:`kOptimValueAsShapeMaxElements`.
inline constexpr int64_t kConstantValueAsShapeMaxElements = kOptimValueAsShapeMaxElements;

/**
 * Computes the output :cpp:class:`SymTensor` of a ``Constant`` node
 * and stores it in ``ctx``.
 *
 * ``Constant`` declares its output as the value of exactly one of the
 * attributes ``value``, ``sparse_value``, ``value_int``,
 * ``value_ints``, ``value_float``, ``value_floats``, ``value_string``
 * or ``value_strings`` (which one is allowed depends on the schema
 * revision, but for shape inference the union of every revision is
 * accepted). The output dtype and shape are taken from the present
 * attribute.
 *
 * When the resulting tensor carries at most
 * :cpp:var:`kConstantValueAsShapeMaxElements` integer elements and
 * has rank at most one — i.e. it is small enough to plausibly be used
 * later as a shape input of operators such as ``Reshape``,
 * ``Expand`` or ``ConstantOfShape`` — its integer values are also
 * recorded via :cpp:func:`SymTensor::SetValueAsShape`. This mirrors
 * the upstream ONNX shape-inference data-propagation behaviour for
 * small integer constants.
 *
 * @param ctx   In/out context. On return contains an entry for
 *              ``node.output(0)`` describing the constant output.
 * @param node  The ``Constant`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Constant"``
 *              and ``node`` must declare at least one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Constant"``, if ``node`` has no output, or if the
 *         attributes do not specify exactly one of the allowed value
 *         forms.
 */
void ComputeShapeConstant(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``ConstantOfShape``
 * node and stores it in ``ctx``.
 *
 * ``ConstantOfShape`` produces a tensor whose shape is given by the
 * single 1-D ``int64`` input and whose element type and fill value are
 * taken from the optional ``value`` attribute (defaults to a single
 * ``float32`` zero).
 *
 * The output element type is inferred from the ``value`` attribute when
 * present, otherwise defaults to :cpp:enumerator:`TensorType::kFloat`.
 * The output shape is taken from the input's
 * :cpp:func:`SymTensor::ValueAsShape` annotation when available. When
 * the input value has not been data-propagated but its static shape is
 * a 1-D tensor whose single dim is a known integer, the corresponding
 * output rank is reconstructed with symbolic dims.
 *
 * @param ctx   In/out context. On return contains an entry for
 *              ``node.output(0)`` describing the output.
 * @param node  The ``ConstantOfShape`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"ConstantOfShape"``, ``node`` must declare exactly one
 *              input and one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"ConstantOfShape"``, if ``node`` has no input or output, or
 *         if the ``value`` attribute is present but does not carry a
 *         tensor value.
 */
void ComputeShapeConstantOfShape(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of an ``EyeLike`` node
 * and stores it in ``ctx``.
 *
 * ``EyeLike`` outputs a tensor with the same 2-D shape as its input.
 * The output dtype is set from the optional ``dtype`` attribute when
 * present, otherwise it defaults to the input dtype.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"EyeLike"``,
 *         if ``node`` has no input or output, if the input rank is not 2, or
 *         if ``dtype`` is present but unsupported.
 */
void ComputeShapeEyeLike(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``BlackmanWindow``
 * node and stores it in ``ctx``.
 *
 * ``BlackmanWindow`` produces a 1-D tensor of length ``size`` (its
 * single scalar integer input) holding the Blackman window
 * coefficients. The output element type is given by the optional
 * ``output_datatype`` attribute (a :cpp:class:`TensorProto::DataType`
 * value, defaults to ``FLOAT``). The ``periodic`` attribute does not
 * affect the output shape or dtype.
 *
 * The output shape is taken from the input's
 * :cpp:func:`SymTensor::ValueAsShape` annotation when available
 * (i.e. when ``size`` is a known constant): the output is then a
 * 1-D tensor with a concrete dim. Otherwise the output is a 1-D
 * tensor with a single symbolic dim.
 *
 * @param ctx   In/out context. On return contains an entry for
 *              ``node.output(0)`` describing the output.
 * @param node  The ``BlackmanWindow`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"BlackmanWindow"``, ``node`` must declare exactly one
 *              input and one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"BlackmanWindow"`` or if ``node`` has no output.
 */
void ComputeShapeBlackmanWindow(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``HannWindow`` node
 * and stores it in ``ctx``. Same semantics as
 * :cpp:func:`ComputeShapeBlackmanWindow` but for the ``HannWindow``
 * operator.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"HannWindow"`` or if ``node`` has no output.
 */
void ComputeShapeHannWindow(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``HammingWindow``
 * node and stores it in ``ctx``. Same semantics as
 * :cpp:func:`ComputeShapeBlackmanWindow` but for the ``HammingWindow``
 * operator.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"HammingWindow"`` or if ``node`` has no output.
 */
void ComputeShapeHammingWindow(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``MelWeightMatrix``
 * node and stores it in ``ctx``.
 *
 * ``MelWeightMatrix`` produces a 2-D tensor of shape
 * ``[floor(dft_length/2) + 1, num_mel_bins]`` holding the triangular
 * Mel filter-bank weights. The output element type is given by the
 * optional ``output_datatype`` attribute (a
 * :cpp:class:`TensorProto::DataType` value, defaults to ``FLOAT``).
 *
 * The output shape is taken from the input's
 * :cpp:func:`SymTensor::ValueAsShape` annotation when available
 * (i.e. when ``num_mel_bins`` and ``dft_length`` are known constants).
 * Otherwise the corresponding output dim is symbolic.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"MelWeightMatrix"`` or if ``node`` has no output.
 */
void ComputeShapeMelWeightMatrix(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``Bernoulli`` node and
 * stores it in ``ctx``.
 *
 * ``Bernoulli`` draws binary samples from a Bernoulli distribution whose
 * probabilities are given by the single input tensor. The output has the
 * same shape as the input. Its element type is given by the optional
 * ``dtype`` attribute (a :cpp:class:`TensorProto::DataType` value); when
 * the attribute is absent, the output dtype matches the input dtype.
 *
 * @param ctx   In/out context. On return contains an entry for
 *              ``node.output(0)`` describing the output.
 * @param node  The ``Bernoulli`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Bernoulli"``,
 *              ``node`` must declare exactly one input and one output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Bernoulli"``, if ``node`` has no input or output, or if the
 *         ``dtype`` attribute is present but holds an unsupported value.
 */
void ComputeShapeBernoulli(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``RandomNormal`` node and
 * stores it in ``ctx``.
 *
 * ``RandomNormal`` produces a tensor whose shape is given by the required
 * ``shape`` attribute (a ``std::vector<int64_t>``) and whose element type is
 * given by the optional ``dtype`` attribute (a :cpp:class:`TensorProto::DataType`
 * value, defaults to ``FLOAT``).
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"RandomNormal"``,
 *         if ``node`` has no output, if the ``shape`` attribute is missing,
 *         contains negative dims, or if ``dtype`` is present but holds an
 *         unsupported value.
 */
void ComputeShapeRandomNormal(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``RandomUniform`` node and
 * stores it in ``ctx``. Same shape/dtype semantics as
 * :cpp:func:`ComputeShapeRandomNormal`.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"RandomUniform"``
 *         or as documented in :cpp:func:`ComputeShapeRandomNormal`.
 */
void ComputeShapeRandomUniform(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``RandomNormalLike``
 * node and stores it in ``ctx``.
 *
 * ``RandomNormalLike`` copies its single input tensor's shape. The output
 * element type is given by the optional ``dtype`` attribute (a
 * :cpp:class:`TensorProto::DataType` value); when the attribute is absent,
 * the output dtype matches the input dtype.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"RandomNormalLike"``, if ``node`` has no input or output, or if
 *         ``dtype`` is present but holds an unsupported value.
 */
void ComputeShapeRandomNormalLike(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``RandomUniformLike``
 * node and stores it in ``ctx``. Same shape/dtype semantics as
 * :cpp:func:`ComputeShapeRandomNormalLike`.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"RandomUniformLike"`` or as documented in
 *         :cpp:func:`ComputeShapeRandomNormalLike`.
 */
void ComputeShapeRandomUniformLike(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``Multinomial`` node
 * and stores it in ``ctx``.
 *
 * ``Multinomial`` draws ``sample_size`` samples per row from a multinomial
 * distribution whose unnormalized log-probabilities are given by the
 * 2-D input tensor of shape ``[batch_size, class_size]``. The output is a
 * 2-D tensor of shape ``[batch_size, sample_size]`` whose element type
 * defaults to ``INT32`` (and may also be ``INT64``).
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Multinomial"``, if ``node`` has no input or output, if the
 *         input rank is statically known to be different from 2, or if
 *         the ``dtype`` attribute is present but is neither ``INT32`` nor
 *         ``INT64``.
 */
void ComputeShapeMultinomial(ShapesContext &ctx, const NodeProto &node);

/**
 * Computes the output :cpp:class:`SymTensor` of a ``Range`` node and
 * stores it in ``ctx``.
 *
 * ``Range`` produces a 1-D tensor whose element type matches the three
 * scalar inputs ``start``, ``limit`` and ``delta`` (which must all share
 * the same dtype, per the schema's ``T`` constraint).
 *
 * When the values of all three inputs have been data-propagated as
 * scalars, the output dimension is explicitly computed as
 * ``max(ceil((limit - start) / delta), 0)``. Otherwise the output is
 * described as a 1-D tensor with a single symbolic dim of unknown size.
 *
 * @param ctx   In/out context. On return contains an entry for
 *              ``node.output(0)`` describing the output.
 * @param node  The ``Range`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Range"`` and
 *              ``node`` must declare exactly three inputs and one
 *              output.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"Range"``,
 *         if ``node`` has fewer than three inputs or no output.
 */
void ComputeShapeRange(ShapesContext &ctx, const NodeProto &node);

} // namespace generator
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes
