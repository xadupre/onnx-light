// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_kernel.h"
#include "onnx_proto/onnx.h"

/**
 * @file abs.h
 * @brief Shape kernel for the ONNX ``Abs`` operator.
 *
 * The ``Abs`` operator is element-wise and unary: its output tensor has
 * exactly the same shape and the same element type as its input. The
 * shape kernel therefore simply forwards the input dtype and shape to
 * the output and produces a fresh :cpp:class:`OptimTensor` whose data
 * pointer is ``nullptr`` (shape inference does not materialise data).
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

/**
 * Shape kernel for the ONNX ``Abs`` operator.
 *
 * The kernel is instantiated from a :cpp:class:`NodeProto` describing
 * the ``Abs`` node, and its :cpp:func:`Run` method propagates the input
 * shape and dtype to the output unchanged.
 */
class AbsShapeKernel : public ShapeKernel {
public:
  /// Constructs the kernel from an ``Abs`` ``NodeProto``.
  explicit AbsShapeKernel(const NodeProto &node);

  /// Returns an :cpp:class:`OptimTensor` whose shape and dtype mirror
  /// those of ``input``. The returned tensor is a pure shape/dtype
  /// description and carries no data buffer.
  OptimTensor Run(const OptimTensor &input) const override;
};

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
