// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

namespace {

OptimDim MulDim(const OptimDim &dim, int64_t factor) {
  if (dim.IsInt()) {
    return OptimDim(dim.AsInt() * factor);
  }
  return OptimDim("(" + dim.AsExpr() + ")*" + std::to_string(factor));
}

OptimDim DivDim(const OptimDim &dim, int64_t divisor, const char *axis_name) {
  if (dim.IsInt()) {
    if (divisor <= 0 || dim.AsInt() % divisor != 0) {
      throw std::invalid_argument(std::string("ComputeShapeSpaceToDepth: input ") + axis_name +
                                  " dim (" + std::to_string(dim.AsInt()) +
                                  ") is not divisible by blocksize (" + std::to_string(divisor) +
                                  ").");
    }
    return OptimDim(dim.AsInt() / divisor);
  }
  return OptimDim("(" + dim.AsExpr() + ")/" + std::to_string(divisor));
}

} // namespace

void ComputeShapeSpaceToDepth(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SpaceToDepth", "ComputeShapeSpaceToDepth");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeSpaceToDepth: SpaceToDepth requires one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const OptimShape &input_shape = input.Shape();

  const AttributeProto *blocksize_attr = FindAttribute(node, "blocksize");
  if (blocksize_attr == nullptr) {
    throw std::invalid_argument(
        "ComputeShapeSpaceToDepth: required attribute 'blocksize' is missing.");
  }
  const int64_t blocksize = blocksize_attr->ref_i();
  if (blocksize <= 0) {
    throw std::invalid_argument("ComputeShapeSpaceToDepth: blocksize must be positive (got " +
                                std::to_string(blocksize) + ").");
  }

  if (input_shape.Rank() != 4) {
    throw std::invalid_argument("ComputeShapeSpaceToDepth: input must be a 4-D tensor (got rank " +
                                std::to_string(input_shape.Rank()) + ").");
  }

  const int64_t bs2 = blocksize * blocksize;
  OptimShape out_shape;
  out_shape.PushBack(input_shape[0]);
  out_shape.PushBack(MulDim(input_shape[1], bs2));
  out_shape.PushBack(DivDim(input_shape[2], blocksize, "H"));
  out_shape.PushBack(DivDim(input_shape[3], blocksize, "W"));

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
