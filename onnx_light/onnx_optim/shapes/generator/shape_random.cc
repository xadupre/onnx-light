// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/generator/shape_generator.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace generator {

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
  if (out_dtype == TensorType::kUndefined) {
    throw std::invalid_argument(std::string(op_name) +
                                ": attribute 'dtype' has unsupported value " +
                                std::to_string(dtype_value) + ".");
  }
  return out_dtype;
}

// Builds the output OptimShape from the ``shape`` attribute (a list of
// non-negative int64 dims). Throws ``std::invalid_argument`` when the
// attribute is missing or contains a negative dim.
OptimShape ShapeFromAttribute(const NodeProto &node, const char *op_name) {
  std::vector<int64_t> dims;
  if (!GetAttributeInts(node, "shape", dims)) {
    throw std::invalid_argument(std::string(op_name) + ": required attribute 'shape' is missing.");
  }
  OptimShape out_shape;
  for (int64_t dim : dims) {
    if (dim < 0) {
      throw std::invalid_argument(std::string(op_name) +
                                  ": attribute 'shape' must not contain negative dims, got " +
                                  std::to_string(dim) + ".");
    }
    out_shape.PushBack(OptimDim(dim));
  }
  return out_shape;
}

void ComputeShapeRandomImpl(ShapesContext &ctx, const NodeProto &node, const char *op_name) {
  const std::string caller = std::string("ComputeShape") + op_name;
  CheckNodeOpAndOutput(node, op_name, caller.c_str());
  TensorType out_dtype = ResolveDtype(node, TensorType::kFloat, caller.c_str());
  OptimShape out_shape = ShapeFromAttribute(node, caller.c_str());
  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
}

void ComputeShapeRandomLikeImpl(ShapesContext &ctx, const NodeProto &node, const char *op_name) {
  const std::string caller = std::string("ComputeShape") + op_name;
  CheckNodeOpAndOutput(node, op_name, caller.c_str());
  EXT_ENFORCE_INVALID(node.input_size() >= 1, caller + ": " + op_name + " requires one input.");
  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  TensorType out_dtype = ResolveDtype(node, input.Dtype(), caller.c_str());
  OptimShape out_shape = input.Shape();
  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
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

} // namespace generator
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
