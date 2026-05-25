// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_kernel.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/math/abs.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

OptimTensor ShapeKernel::Run(const OptimTensor & /*input*/) const {
  throw std::logic_error("ShapeKernel::Run(OptimTensor) not implemented for op_type='" + op_type_ +
                         "' (domain='" + domain_ + "').");
}

namespace {

bool IsOnnxDomain(const std::string &domain) { return domain.empty() || domain == "ai.onnx"; }

} // namespace

std::unique_ptr<ShapeKernel> MakeShapeKernel(const NodeProto &node) {
  const std::string op_type = node.op_type().as_string();
  const std::string domain = node.domain().as_string();

  if (IsOnnxDomain(domain)) {
    if (op_type == "Abs") {
      return std::make_unique<math::AbsShapeKernel>(node);
    }
  }

  throw std::runtime_error("No shape kernel registered for op_type='" + op_type + "' (domain='" +
                           domain + "').");
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
