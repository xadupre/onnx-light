// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_check.h"
#include "onnx_extensions/shapes/shapes/math/shape_math.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

void ComputeShapeGeGLU(ShapesContext &ctx, const NodeProto &node, const char *a, const char *b) {
  CheckNodeOpAndOutput(node, "GeGLU", "ComputeShapeGeGLU");
  const AttributeProto *attr = FindAttribute(node, "approximate");
  EXT_ENFORCE_INVALID(attr == nullptr || attr->type() == AttributeProto::STRING,
                      "ComputeShapeGeGLU: 'approximate' must be a STRING.");
  const std::string approximate = attr == nullptr ? "none" : attr->s();
  EXT_ENFORCE_INVALID(approximate == "none" || approximate == "tanh",
                      "ComputeShapeGeGLU: 'approximate' must be 'none' or 'tanh'.");
  const SymTensor &gate = ctx.Get(a);
  const SymTensor &value = ctx.Get(b);
  const auto dtype = gate.Dtype();
  EXT_ENFORCE_INVALID(dtype == TensorType::kFloat || dtype == TensorType::kDouble ||
                          dtype == TensorType::kFloat16 || dtype == TensorType::kBfloat16,
                      "ComputeShapeGeGLU: unsupported input dtype.");
  EXT_ENFORCE_INVALID(dtype == value.Dtype(), "ComputeShapeGeGLU: input dtypes must match.");
  auto shape = gate.Shape();
  const auto &other = value.Shape();
  EXT_ENFORCE_INVALID(shape.Rank() == other.Rank(),
                      "ComputeShapeGeGLU: input ranks must match (no broadcasting).");
  for (size_t i = 0; i < shape.Rank(); ++i) {
    EXT_ENFORCE_INVALID(!shape[i].IsInt() || !other[i].IsInt() || shape[i] == other[i],
                        "ComputeShapeGeGLU: input dimensions must match (no broadcasting).");
    if (!shape[i].IsInt() && other[i].IsInt()) {
      shape[i] = other[i];
    } else if (shape[i].IsExpr() && shape[i].AsExpr().empty()) {
      shape[i] = other[i];
    }
  }
  ctx.Set(node.output(0), SymTensor(nullptr, dtype, shape));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
