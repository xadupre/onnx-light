// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

namespace {

SymDim MulDim(const SymDim &dim, int64_t factor) {
  if (dim.IsInt()) {
    return SymDim(dim.AsInt() * factor);
  }
  return SymDim("(" + dim.AsExpr() + ")*" + std::to_string(factor));
}

SymDim DivDim(const SymDim &dim, int64_t divisor) {
  if (dim.IsInt()) {
    EXT_ENFORCE_INVALID(!(divisor <= 0 || dim.AsInt() % divisor != 0),
                        "ComputeShapeDepthToSpace: input channel dim (", dim.AsInt(),
                        ") is not divisible by blocksize * blocksize (", divisor, ").");
    return SymDim(dim.AsInt() / divisor);
  }
  return SymDim("(" + dim.AsExpr() + ")/" + std::to_string(divisor));
}

} // namespace

void ComputeShapeDepthToSpace(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "DepthToSpace", "ComputeShapeDepthToSpace");
  EXT_ENFORCE_INVALID(!(node.input_size() < 1),
                      "ComputeShapeDepthToSpace: DepthToSpace requires one input.");

  const SymTensor &input = ctx.Get(node.input(0));
  const SymShape &input_shape = input.Shape();

  int64_t blocksize = 0;
  const AttributeProto *blocksize_attr = FindAttribute(node, "blocksize");
  EXT_ENFORCE_INVALID(blocksize_attr != nullptr,
                      "ComputeShapeDepthToSpace: required attribute 'blocksize' is missing.");
  blocksize = blocksize_attr->ref_i();
  EXT_ENFORCE_INVALID(!(blocksize <= 0),
                      "ComputeShapeDepthToSpace: blocksize must be positive (got ", blocksize,
                      ").");

  EXT_ENFORCE_INVALID(input_shape.Rank() == 4,
                      "ComputeShapeDepthToSpace: input must be a 4-D tensor (got rank ",
                      input_shape.Rank(), ").");

  const int64_t bs2 = blocksize * blocksize;
  SymShape out_shape;
  out_shape.PushBack(input_shape[0]);
  out_shape.PushBack(DivDim(input_shape[1], bs2));
  out_shape.PushBack(MulDim(input_shape[2], blocksize));
  out_shape.PushBack(MulDim(input_shape[3], blocksize));

  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
