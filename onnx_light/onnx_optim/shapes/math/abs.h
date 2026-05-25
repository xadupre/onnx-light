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
 * the ``Abs`` node together with the opset version of the ``ai.onnx``
 * domain under which the node should be interpreted. Its
 * :cpp:func:`Run` method propagates the input shape and dtype to the
 * output unchanged.
 *
 * ``Abs`` is element-wise in every revision of its schema (since
 * version 1, with subsequent updates at 6 and 13 only widening the
 * accepted dtype set), so shape inference is identical for every
 * supported opset; the opset version is nevertheless validated and
 * exposed via :cpp:func:`SinceVersion` for symmetry with other shape
 * kernels.
 */
class AbsShapeKernel : public ShapeKernel {
public:
  /// Earliest ``ai.onnx`` opset version supported by this kernel.
  static constexpr int kMinOpsetVersion = 1;
  /// Latest ``ai.onnx`` opset revision known to this kernel.
  static constexpr int kLatestSinceVersion = 13;

  /// Resolves the ``Abs`` schema ``since_version`` matching
  /// ``opset_version``. Passing :cpp:var:`kUnknownOpsetVersion`
  /// selects :cpp:var:`kLatestSinceVersion`.
  static int ResolveSinceVersion(int opset_version);

  /// Constructs the kernel from an ``Abs`` ``NodeProto``.
  ///
  /// @param node           The ``NodeProto`` whose ``op_type`` must be
  ///                       ``"Abs"``.
  /// @param opset_version  Opset version of ``ai.onnx`` declared by the
  ///                       surrounding model. Pass
  ///                       :cpp:var:`kUnknownOpsetVersion` to default
  ///                       to the latest supported revision.
  explicit AbsShapeKernel(const NodeProto &node, int opset_version = kUnknownOpsetVersion);

  /// Returns an :cpp:class:`OptimTensor` whose shape and dtype mirror
  /// those of ``input``. The returned tensor is a pure shape/dtype
  /// description and carries no data buffer.
  OptimTensor Run(const OptimTensor &input) const override;
};

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
