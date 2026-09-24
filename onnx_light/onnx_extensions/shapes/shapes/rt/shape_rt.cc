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

void ComputeShapeQuantize(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Quantize", "ComputeShapeQuantize");
  EXT_ENFORCE_INVALID(node.input_size() >= 1 && node.input_size() <= 9 && node.output_size() == 1,
                      "Quantize requires 1 to 9 inputs and one output.");
  const auto *attribute = FindAttribute(node, "type");
  EXT_ENFORCE_INVALID(attribute && attribute->type() == AttributeProto::TYPE_PROTO &&
                          attribute->has_tp() && attribute->tp().has_struct_type(),
                      "Quantize requires a 'type' attribute containing StructTypeProto.");
  ctx.SetType(node.output(0), attribute->tp());
}

void ComputeShapeDequantize(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Dequantize", "ComputeShapeDequantize");
  EXT_ENFORCE_INVALID(node.input_size() == 1 && node.output_size() == 1,
                      "Dequantize requires one input and one output.");
  const int64_t dtype = RequiredIntAttributeValue(node, "dtype", "Dequantize");
  EXT_ENFORCE_INVALID(dtype == TensorProto::FLOAT || dtype == TensorProto::DOUBLE ||
                          dtype == TensorProto::FLOAT16 || dtype == TensorProto::BFLOAT16,
                      "Dequantize dtype must be FLOAT, DOUBLE, FLOAT16 or BFLOAT16.");
  if (ctx.HasType(node.input(0)))
    EXT_ENFORCE_INVALID(ctx.GetType(node.input(0)).has_struct_type(),
                        "Dequantize requires a structured encoded input.");
  TypeProto output;
  if (ctx.HasEncodedValue(node.input(0))) {
    const auto &encoded = ctx.GetEncodedValue(node.input(0));
    EXT_ENFORCE_INVALID(encoded.has_logical_type() && encoded.logical_type().has_tensor_type(),
                        "Dequantize input requires a logical tensor type.");
    output = encoded.logical_type();
  }
  output.mutable_tensor_type()->set_elem_type(static_cast<int32_t>(dtype));
  ctx.SetType(node.output(0), output);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::rt
