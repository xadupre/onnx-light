// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/abs.h"

#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

int AbsShapeKernel::ResolveSinceVersion(int opset_version) {
  if (opset_version == kUnknownOpsetVersion) {
    return kLatestSinceVersion;
  }
  // Abs schema history in ai.onnx: v1, v6, v13.
  if (opset_version >= 13) {
    return 13;
  }
  if (opset_version >= 6) {
    return 6;
  }
  return 1;
}

AbsShapeKernel::AbsShapeKernel(const NodeProto &node, int opset_version)
    : ShapeKernel(node, opset_version, ResolveSinceVersion(opset_version)) {
  if (node.op_type().as_string() != "Abs") {
    throw std::invalid_argument("AbsShapeKernel expects op_type='Abs', got '" +
                                node.op_type().as_string() + "'.");
  }
  if (opset_version != kUnknownOpsetVersion && opset_version < kMinOpsetVersion) {
    throw std::runtime_error(
        "AbsShapeKernel does not support opset_version=" + std::to_string(opset_version) +
        " (minimum supported: " + std::to_string(kMinOpsetVersion) + ").");
  }
}

OptimTensor AbsShapeKernel::Run(const OptimTensor &input) const {
  // Abs is element-wise in every supported opset revision: the output
  // dtype and shape match the input.
  return OptimTensor(nullptr, input.Dtype(), input.Shape());
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
