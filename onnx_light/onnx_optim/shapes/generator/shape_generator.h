// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

/**
 * @file shape_generator.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``generator`` family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace generator {

/// Maximum element count of a ``Constant`` value tensor for which
/// :cpp:func:`ComputeShapeConstant` populates the output
/// :cpp:func:`OptimTensor::ValueAsShape` annotation. Constants beyond
/// this threshold are not data-propagated (the output dtype and shape
/// are still inferred normally).
inline constexpr int64_t kConstantValueAsShapeMaxElements = 8;

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Constant`` node
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
 * recorded via :cpp:func:`OptimTensor::SetValueAsShape`. This mirrors
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
 * Computes the output :cpp:class:`OptimTensor` of a ``ConstantOfShape``
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
 * :cpp:func:`OptimTensor::ValueAsShape` annotation when available. When
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
 * Computes the output :cpp:class:`OptimTensor` of a ``BlackmanWindow``
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
 * :cpp:func:`OptimTensor::ValueAsShape` annotation when available
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
 * Computes the output :cpp:class:`OptimTensor` of a ``Bernoulli`` node and
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

} // namespace generator
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
