// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/generator/shape_generator.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::generator {

namespace {

// Resolves the output dtype from the optional ``dtype`` attribute. When
// the attribute is absent, returns ``default_dtype``. Throws
// ``std::invalid_argument`` when ``dtype`` is present but does not name a
// supported tensor element type.
TensorType ResolveDtype(const NodeProto &node, TensorType default_dtype, const char *op_name) {
  const AttributeProto *dtype_attr = FindAttribute(node, "dtype");
  if (dtype_attr == nullptr) {
    return default_dtype;
  }
  const int64_t dtype_value = dtype_attr->i();
  TensorType out_dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype_value));
  EXT_ENFORCE_INVALID(out_dtype != TensorType::kUndefined, op_name,
                      ": attribute 'dtype' has unsupported value ", dtype_value, ".");
  return out_dtype;
}

// Builds the output SymShape from the ``shape`` attribute (a list of
// non-negative int64 dims). Throws ``std::invalid_argument`` when the
// attribute is missing or contains a negative dim.
SymShape ShapeFromAttribute(const NodeProto &node, const char *op_name) {
  std::vector<int64_t> dims;
  EXT_ENFORCE_INVALID(GetAttributeInts(node, "shape", dims), op_name,
                      ": required attribute 'shape' is missing.");
  SymShape out_shape;
  for (int64_t dim : dims) {
    EXT_ENFORCE_INVALID(!(dim < 0), op_name,
                        ": attribute 'shape' must not contain negative dims, got ", dim, ".");
    out_shape.PushBack(SymDim(dim));
  }
  return out_shape;
}

void ComputeShapeRandomImpl(ShapesContext &ctx, const NodeProto &node, const char *op_name) {
  const std::string caller = std::string("ComputeShape") + op_name;
  CheckNodeOpAndOutput(node, op_name, caller.c_str());
  TensorType out_dtype = ResolveDtype(node, TensorType::kFloat, caller.c_str());
  SymShape out_shape = ShapeFromAttribute(node, caller.c_str());
  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

void ComputeShapeRandomLikeImpl(ShapesContext &ctx, const NodeProto &node, const char *op_name) {
  const std::string caller = std::string("ComputeShape") + op_name;
  CheckNodeOpAndOutput(node, op_name, caller.c_str());
  EXT_ENFORCE_INVALID(node.input_size() >= 1, caller, ": ", op_name, " requires one input.");
  const SymTensor &input = ctx.Get(node.input(0));
  TensorType out_dtype = ResolveDtype(node, input.Dtype(), caller.c_str());
  SymShape out_shape = input.Shape();
  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace

void ComputeShapeRandomNormal(ShapesContext &ctx, const NodeProto &node) {
  ComputeShapeRandomImpl(ctx, node, "RandomNormal");
}

void ComputeShapeRandomUniform(ShapesContext &ctx, const NodeProto &node) {
  ComputeShapeRandomImpl(ctx, node, "RandomUniform");
}

void ComputeShapeRandomNormalLike(ShapesContext &ctx, const NodeProto &node) {
  ComputeShapeRandomLikeImpl(ctx, node, "RandomNormalLike");
}

void ComputeShapeRandomUniformLike(ShapesContext &ctx, const NodeProto &node) {
  ComputeShapeRandomLikeImpl(ctx, node, "RandomUniformLike");
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::generator
