// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ValidateScatterReduction(const ShapesContext &ctx, const NodeProto &node,
                              const SymTensor &data) {
  const int opset = ctx.OpsetVersion(node.domain());
  if (opset > 0 && opset < 29) {
    return;
  }
  const auto reduction = GetAttributeOr<std::string>(node, "reduction", "none");
  EXT_ENFORCE_INVALID(reduction == "none" || reduction == "add" || reduction == "mul" ||
                          reduction == "max" || reduction == "min" || reduction == "mean",
                      "Invalid scatter reduction: ", reduction);
  if (reduction == "mean") {
    const auto dtype = data.Dtype();
    EXT_ENFORCE_INVALID(dtype != TensorType::kBool && dtype != TensorType::kString &&
                            dtype != TensorType::kComplex64 && dtype != TensorType::kComplex128,
                        "Scatter reduction 'mean' requires an integer or floating-point tensor.");
  }
}

void ComputeShapeScatterElements(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "ScatterElements", "ComputeShapeScatterElements");

  EXT_ENFORCE_INVALID(!(node.input_size() < 3),
                      "ComputeShapeScatterElements: ScatterElements requires three "
                      "inputs (data, indices, updates).");

  const SymTensor &data = ctx.Get(node.input(0));
  ValidateScatterReduction(ctx, node, data);

  // Output has the same shape and dtype as data.
  ctx.Set(node.output(0), SymTensor(nullptr, data.Dtype(), data.Shape()));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
