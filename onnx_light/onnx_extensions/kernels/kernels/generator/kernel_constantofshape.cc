// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Returns true if ``dtype`` is one of the numeric (non-STRING) element
// types whose byte-stride matches ``ElementSize(dtype)`` and which can
// therefore be broadcast by simple buffer fills.
bool IsSupportedConstantOfShapeDtype(int32_t dtype) {
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT:
  case DataType::DOUBLE:
  case DataType::INT8:
  case DataType::UINT8:
  case DataType::INT16:
  case DataType::UINT16:
  case DataType::INT32:
  case DataType::UINT32:
  case DataType::INT64:
  case DataType::UINT64:
  case DataType::BOOL:
  case DataType::FLOAT16:
  case DataType::BFLOAT16:
    return true;
  default:
    return false;
  }
}

// Reads the output shape from the 1-D INT64 ``shape`` input tensor.
Shape ReadShapeInput(const Tensor &shape) {
  EXT_ENFORCE_INVALID(shape.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ConstantOfShape: 'shape' input must be INT64.");
  EXT_ENFORCE_INVALID(shape.shape.size() <= 1,
                      "kernel::ConstantOfShape: 'shape' input must be a 1-D tensor.");
  if (shape.shape.empty()) {
    // 0-D scalar: treat as length-zero, producing a scalar output.
    return Shape{};
  }
  const int64_t n = shape.shape[0];
  Shape out;
  const int64_t *dims = shape.AsInt64();
  if (n > 0) {
    out.insert(out.end(), dims, dims + n);
  }
  for (int64_t d : out) {
    EXT_ENFORCE_INVALID(d >= 0, "kernel::ConstantOfShape: shape entries must be non-negative.");
  }
  return out;
}

} // namespace

Tensor ConstantOfShape::operator()(const Tensor &shape, const Tensor &value,
                                   RuntimeContext *rt) const {
  const Shape out_shape = ReadShapeInput(shape);

  // Default fill: a single FLOAT 0.0 (per the ONNX schema).
  int32_t out_dtype = value.data_type;
  std::vector<uint8_t> elem_bytes;
  if (value.size_bytes() == 0 && value.data_type == 0) {
    out_dtype = static_cast<int32_t>(DataType::FLOAT);
    elem_bytes.assign(sizeof(float), 0);
  } else {
    EXT_ENFORCE_INVALID(IsSupportedConstantOfShapeDtype(value.data_type),
                        "kernel::ConstantOfShape: unsupported 'value' dtype.");
    const std::size_t es = ElementSize(value.data_type);
    EXT_ENFORCE_INVALID(value.size_bytes() == es,
                        "kernel::ConstantOfShape: 'value' attribute must hold exactly one "
                        "element.");
    elem_bytes.assign(value.bytes(), value.bytes() + value.size_bytes());
  }

  const int64_t n = out_shape.product();
  const std::size_t es = elem_bytes.size();
  Tensor out = MakeOutputTensor(out_dtype, out_shape, static_cast<std::size_t>(n) * es,
                                rt ? rt->allocator() : nullptr);
  uint8_t *out_ptr = out.mutable_bytes();
  for (int64_t i = 0; i < n; ++i) {
    std::memcpy(out_ptr + static_cast<std::size_t>(i) * es, elem_bytes.data(), es);
  }
  return out;
}

void ConstantOfShape::operator()(const Tensor &shape, const Tensor &value, Tensor &output) const {
  Tensor produced = (*this)(shape, value);
  EXT_ENFORCE_INVALID(output.data_type == produced.data_type,
                      "kernel::ConstantOfShape preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == produced.shape,
                      "kernel::ConstantOfShape preallocated output shape must match the produced "
                      "tensor shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == produced.size_bytes(),
                      "kernel::ConstantOfShape preallocated output buffer has unexpected size in "
                      "bytes.");
  if (!produced.data.empty()) {
    std::memcpy(output.mutable_bytes(), produced.bytes(), produced.size_bytes());
  }
}

void ConstantOfShape::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &shape = GetInput(node, 0, rt.tensors());
  Tensor value;
  if (FindAttribute(node, "value") != nullptr) {
    value = GetRequiredAttributeTensor(node, "value");
  }
  onnx_kernels::kernel::ConstantOfShape k(rt.kernel_ctx());
  SetOutput(node, 0, k(shape, value, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
