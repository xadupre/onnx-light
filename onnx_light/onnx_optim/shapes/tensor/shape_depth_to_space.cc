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

OptimDim DivDim(const OptimDim &dim, int64_t divisor) {
  if (dim.IsInt()) {
    if (divisor <= 0 || dim.AsInt() % divisor != 0) {
      throw std::invalid_argument(
          "ComputeShapeDepthToSpace: input channel dim (" + std::to_string(dim.AsInt()) +
          ") is not divisible by blocksize * blocksize (" + std::to_string(divisor) + ").");
    }
    return OptimDim(dim.AsInt() / divisor);
  }
  return OptimDim("(" + dim.AsExpr() + ")/" + std::to_string(divisor));
}

} // namespace

void ComputeShapeDepthToSpace(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "DepthToSpace", "ComputeShapeDepthToSpace");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeDepthToSpace: DepthToSpace requires one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const OptimShape &input_shape = input.Shape();

  int64_t blocksize = 0;
  const AttributeProto *blocksize_attr = FindAttribute(node, "blocksize");
  if (blocksize_attr == nullptr) {
    throw std::invalid_argument(
        "ComputeShapeDepthToSpace: required attribute 'blocksize' is missing.");
  }
  blocksize = blocksize_attr->ref_i();
  if (blocksize <= 0) {
    throw std::invalid_argument("ComputeShapeDepthToSpace: blocksize must be positive (got " +
                                std::to_string(blocksize) + ").");
  }

  if (input_shape.Rank() != 4) {
    throw std::invalid_argument("ComputeShapeDepthToSpace: input must be a 4-D tensor (got rank " +
                                std::to_string(input_shape.Rank()) + ").");
  }

  const int64_t bs2 = blocksize * blocksize;
  OptimShape out_shape;
  out_shape.PushBack(input_shape[0]);
  out_shape.PushBack(DivDim(input_shape[1], bs2));
  out_shape.PushBack(MulDim(input_shape[2], blocksize));
  out_shape.PushBack(MulDim(input_shape[3], blocksize));

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
