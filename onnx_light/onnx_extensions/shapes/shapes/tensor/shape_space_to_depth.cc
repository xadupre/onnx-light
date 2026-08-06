// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
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

SymDim DivDim(const SymDim &dim, int64_t divisor, const char *axis_name) {
  if (dim.IsInt()) {
    EXT_ENFORCE_INVALID(!(divisor <= 0 || dim.AsInt() % divisor != 0),
                        "ComputeShapeSpaceToDepth: input ", axis_name, " dim (", dim.AsInt(),
                        ") is not divisible by blocksize (", divisor, ").");
    return SymDim(dim.AsInt() / divisor);
  }
  return SymDim("(" + dim.AsExpr() + ")/" + std::to_string(divisor));
}

} // namespace

void ComputeShapeSpaceToDepth(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SpaceToDepth", "ComputeShapeSpaceToDepth");
  EXT_ENFORCE_INVALID(!(node.input_size() < 1),
                      "ComputeShapeSpaceToDepth: SpaceToDepth requires one input.");

  const SymTensor &input = ctx.Get(node.input(0));
  const SymShape &input_shape = input.Shape();

  const AttributeProto *blocksize_attr = FindAttribute(node, "blocksize");
  EXT_ENFORCE_INVALID(blocksize_attr != nullptr,
                      "ComputeShapeSpaceToDepth: required attribute 'blocksize' is missing.");
  const int64_t blocksize = blocksize_attr->ref_i();
  EXT_ENFORCE_INVALID(!(blocksize <= 0),
                      "ComputeShapeSpaceToDepth: blocksize must be positive (got ", blocksize,
                      ").");

  EXT_ENFORCE_INVALID(input_shape.Rank() == 4,
                      "ComputeShapeSpaceToDepth: input must be a 4-D tensor (got rank ",
                      input_shape.Rank(), ").");

  const int64_t bs2 = blocksize * blocksize;
  SymShape out_shape;
  out_shape.PushBack(input_shape[0]);
  out_shape.PushBack(MulDim(input_shape[1], bs2));
  out_shape.PushBack(DivDim(input_shape[2], blocksize, "H"));
  out_shape.PushBack(DivDim(input_shape[3], blocksize, "W"));

  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
