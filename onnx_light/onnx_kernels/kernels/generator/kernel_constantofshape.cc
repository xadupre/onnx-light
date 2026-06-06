// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/generator/include_generator_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

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

// Computes the total element count from a shape vector (1 for an empty shape).
int64_t ProductOfShape(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0, "kernel::ConstantOfShape: shape entries must be non-negative.");
    n *= d;
  }
  return n;
}

// Reads the output shape from the 1-D INT64 ``shape`` input tensor.
std::vector<int64_t> ReadShapeInput(const Tensor &shape) {
  EXT_ENFORCE_INVALID(shape.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ConstantOfShape: 'shape' input must be INT64.");
  EXT_ENFORCE_INVALID(shape.shape.size() <= 1,
                      "kernel::ConstantOfShape: 'shape' input must be a 1-D tensor.");
  if (shape.shape.empty()) {
    // 0-D scalar: treat as length-zero, producing a scalar output.
    return {};
  }
  const int64_t n = shape.shape[0];
  std::vector<int64_t> out(static_cast<std::size_t>(n));
  if (n > 0) {
    std::memcpy(out.data(), shape.data.data(), static_cast<std::size_t>(n) * sizeof(int64_t));
  }
  return out;
}

} // namespace

Tensor ConstantOfShape::operator()(const Tensor &shape, const Tensor &value) const {
  const std::vector<int64_t> out_shape = ReadShapeInput(shape);

  // Default fill: a single FLOAT 0.0 (per the ONNX schema).
  int32_t out_dtype = value.data_type;
  std::vector<uint8_t> elem_bytes;
  if (value.data.empty() && value.data_type == 0) {
    out_dtype = static_cast<int32_t>(DataType::FLOAT);
    elem_bytes.assign(sizeof(float), 0);
  } else {
    EXT_ENFORCE_INVALID(IsSupportedConstantOfShapeDtype(value.data_type),
                        "kernel::ConstantOfShape: unsupported 'value' dtype.");
    const std::size_t es = ElementSize(value.data_type);
    EXT_ENFORCE_INVALID(value.data.size() == es,
                        "kernel::ConstantOfShape: 'value' attribute must hold exactly one "
                        "element.");
    elem_bytes.assign(value.data.begin(), value.data.end());
  }

  const int64_t n = ProductOfShape(out_shape);
  const std::size_t es = elem_bytes.size();
  std::vector<uint8_t> out_data(static_cast<std::size_t>(n) * es);
  for (int64_t i = 0; i < n; ++i) {
    std::memcpy(out_data.data() + static_cast<std::size_t>(i) * es, elem_bytes.data(), es);
  }
  return Tensor("", out_dtype, out_shape, std::move(out_data));
}

void ConstantOfShape::operator()(const Tensor &shape, const Tensor &value, Tensor &output) const {
  Tensor produced = (*this)(shape, value);
  EXT_ENFORCE_INVALID(output.data_type == produced.data_type,
                      "kernel::ConstantOfShape preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == produced.shape,
                      "kernel::ConstantOfShape preallocated output shape must match the produced "
                      "tensor shape.");
  EXT_ENFORCE_INVALID(output.data.size() == produced.data.size(),
                      "kernel::ConstantOfShape preallocated output buffer has unexpected size in "
                      "bytes.");
  if (!produced.data.empty()) {
    std::memcpy(output.data.data(), produced.data.data(), produced.data.size());
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
