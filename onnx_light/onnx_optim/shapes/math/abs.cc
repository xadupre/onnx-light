// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/abs.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace math {

AbsShapeKernel::AbsShapeKernel(const NodeProto &node) : ShapeKernel(node) {
  if (node.op_type().as_string() != "Abs") {
    throw std::invalid_argument("AbsShapeKernel expects op_type='Abs', got '" +
                                node.op_type().as_string() + "'.");
  }
}

OptimTensor AbsShapeKernel::Run(const OptimTensor &input) const {
  // Abs is element-wise: the output dtype and shape match the input.
  return OptimTensor(nullptr, input.Dtype(), input.Shape());
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
