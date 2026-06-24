// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/rt/shape_rt.h"

#include <string>
#include <vector>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace rt {

namespace {

TensorType ResolveDtype(const NodeProto &node, const char *op_name) {
  const AttributeProto *dtype_attr = FindAttribute(node, "dtype");
  EXT_ENFORCE_INVALID(dtype_attr != nullptr, op_name, ": required attribute 'dtype' is missing.");
  const int64_t dtype_value = dtype_attr->i();
  TensorType out_dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype_value));
  EXT_ENFORCE_INVALID(out_dtype != TensorType::kUndefined, op_name,
                      ": attribute 'dtype' has unsupported value ", dtype_value, ".");
  return out_dtype;
}

OptimShape ShapeFromAttribute(const NodeProto &node, const char *op_name) {
  std::vector<int64_t> dims;
  EXT_ENFORCE_INVALID(GetAttributeInts(node, "shape", dims), op_name,
                      ": required attribute 'shape' is missing.");
  OptimShape out_shape;
  for (int64_t dim : dims) {
    EXT_ENFORCE_INVALID(!(dim < 0), op_name,
                        ": attribute 'shape' must not contain negative dims, got ", dim, ".");
    out_shape.PushBack(OptimDim(dim));
  }
  return out_shape;
}

} // namespace

void ComputeShapeDelayedInitializer(ShapesContext &ctx, const NodeProto &node) {
  constexpr const char *kCaller = "ComputeShapeDelayedInitializer";
  CheckNodeOpAndOutput(node, "DelayedInitializer", kCaller);
  EXT_ENFORCE_INVALID(node.input_size() == 0, kCaller, ": DelayedInitializer requires no inputs.");
  TensorType out_dtype = ResolveDtype(node, kCaller);
  OptimShape out_shape = ShapeFromAttribute(node, kCaller);
  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace rt
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
