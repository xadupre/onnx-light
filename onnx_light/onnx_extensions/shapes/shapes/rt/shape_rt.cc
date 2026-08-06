// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/rt/shape_rt.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::rt {

namespace {

TensorType ResolveDtype(const NodeProto &node, const char *op_name) {
  const AttributeProto *dtype_attr = FindAttribute(node, "dtype");
  EXT_ENFORCE_INVALID(dtype_attr != nullptr, op_name, ": required attribute 'dtype' is missing.");
  const int64_t dtype_value = dtype_attr->i();
  EXT_ENFORCE_INVALID(dtype_value >= std::numeric_limits<int32_t>::min() &&
                          dtype_value <= std::numeric_limits<int32_t>::max(),
                      op_name, ": attribute 'dtype' is out of range: ", dtype_value, ".");
  TensorType out_dtype = DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype_value));
  EXT_ENFORCE_INVALID(out_dtype != TensorType::kUndefined, op_name,
                      ": attribute 'dtype' has unsupported value ", dtype_value, ".");
  EXT_ENFORCE_INVALID(out_dtype != TensorType::kString, op_name,
                      ": attribute 'dtype' does not support STRING tensors.");
  return out_dtype;
}

SymShape ShapeFromAttribute(const NodeProto &node, const char *op_name) {
  std::vector<int64_t> dims;
  EXT_ENFORCE_INVALID(GetAttributeInts(node, "shape", dims), op_name,
                      ": required attribute 'shape' is missing.");
  SymShape out_shape;
  for (int64_t dim : dims) {
    EXT_ENFORCE_INVALID(dim >= 0, op_name,
                        ": attribute 'shape' must not contain negative dims, got ", dim, ".");
    out_shape.PushBack(SymDim(dim));
  }
  return out_shape;
}

std::string RequiredStringAttributeValue(const NodeProto &node, const char *name,
                                         const char *op_name) {
  const AttributeProto *attr = FindAttribute(node, name);
  EXT_ENFORCE_INVALID(attr != nullptr, op_name, ": required attribute '", name, "' is missing.");
  EXT_ENFORCE_INVALID(attr->type() == AttributeProto::AttributeType::STRING, op_name,
                      ": attribute '", name, "' must be STRING.");
  return attr->s();
}

int64_t RequiredIntAttributeValue(const NodeProto &node, const char *name, const char *op_name) {
  const AttributeProto *attr = FindAttribute(node, name);
  EXT_ENFORCE_INVALID(attr != nullptr, op_name, ": required attribute '", name, "' is missing.");
  EXT_ENFORCE_INVALID(attr->type() == AttributeProto::AttributeType::INT, op_name, ": attribute '",
                      name, "' must be INT.");
  return attr->i();
}

void ValidateDeviceAttributes(const NodeProto &node, const char *op_name) {
  const std::string load_device = RequiredStringAttributeValue(node, "load_device", op_name);
  EXT_ENFORCE_INVALID(load_device == "cpu" || load_device == "file", op_name,
                      ": attribute 'load_device' must be 'cpu' or 'file', got '", load_device,
                      "'.");
  const std::string runtime_device = RequiredStringAttributeValue(node, "runtime_device", op_name);
  EXT_ENFORCE_INVALID(runtime_device == "cpu", op_name,
                      ": attribute 'runtime_device' must be 'cpu', got '", runtime_device, "'.");
  const std::string filename = RequiredStringAttributeValue(node, "filename", op_name);
  EXT_ENFORCE_INVALID(!filename.empty(), op_name, ": attribute 'filename' must not be empty.");
  const int64_t offset = RequiredIntAttributeValue(node, "offset", op_name);
  EXT_ENFORCE_INVALID(offset >= 0, op_name, ": attribute 'offset' must be non-negative, got ",
                      offset, ".");
}

} // namespace

void ComputeShapeDelayedInitializer(ShapesContext &ctx, const NodeProto &node) {
  constexpr const char *kCaller = "ComputeShapeDelayedInitializer";
  CheckNodeOpAndOutput(node, "DelayedInitializer", kCaller);
  EXT_ENFORCE_INVALID(node.input_size() == 0, kCaller, ": DelayedInitializer requires no inputs.");
  TensorType out_dtype = ResolveDtype(node, kCaller);
  SymShape out_shape = ShapeFromAttribute(node, kCaller);
  ValidateDeviceAttributes(node, kCaller);
  ctx.Set(node.output(0), SymTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::rt
