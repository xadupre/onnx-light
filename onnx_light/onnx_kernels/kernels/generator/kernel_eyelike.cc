// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/generator/include_generator_kernels.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

bool IsSupportedEyeLikeDtype(int32_t dtype) {
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

std::vector<uint8_t> OneElementBytes(int32_t dtype) {
  const std::size_t es = ElementSize(dtype);
  std::vector<uint8_t> one(es, uint8_t{0});
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT: {
    const float v = 1.0f;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::DOUBLE: {
    const double v = 1.0;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::INT8: {
    const int8_t v = 1;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::UINT8: {
    const uint8_t v = 1;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::INT16: {
    const int16_t v = 1;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::UINT16: {
    const uint16_t v = 1;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::INT32: {
    const int32_t v = 1;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::UINT32: {
    const uint32_t v = 1;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::INT64: {
    const int64_t v = 1;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::UINT64: {
    const uint64_t v = 1;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::BOOL: {
    const uint8_t v = 1;
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::FLOAT16: {
    const uint16_t v = 0x3C00; // IEEE 754 binary16 encoding of 1.0
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  case DataType::BFLOAT16: {
    const uint16_t v = 0x3F80; // bfloat16 encoding of 1.0
    std::memcpy(one.data(), &v, sizeof(v));
    return one;
  }
  default:
    throw std::invalid_argument("kernel::EyeLike: unsupported output dtype.");
  }
}

} // namespace

Tensor EyeLike::operator()(const Tensor &input, int64_t k, int32_t dtype) const {
  EXT_ENFORCE_INVALID(input.shape.size() == 2, "kernel::EyeLike: input must be 2-dimensional.");
  const int64_t rows = input.shape[0];
  const int64_t cols = input.shape[1];
  EXT_ENFORCE_INVALID(rows >= 0 && cols >= 0,
                      "kernel::EyeLike: input dimensions must be non-negative.");

  const int32_t out_dtype = (dtype != 0) ? dtype : input.data_type;
  EXT_ENFORCE_INVALID(IsSupportedEyeLikeDtype(out_dtype),
                      "kernel::EyeLike: unsupported output dtype.");
  const std::size_t es = ElementSize(out_dtype);
  std::vector<uint8_t> out_data(static_cast<std::size_t>(rows * cols) * es, uint8_t{0});
  const std::vector<uint8_t> one = OneElementBytes(out_dtype);
  for (int64_t i = 0; i < rows; ++i) {
    const int64_t j = i + k;
    if (j >= 0 && j < cols) {
      const std::size_t offset = static_cast<std::size_t>(i * cols + j) * es;
      std::memcpy(out_data.data() + offset, one.data(), es);
    }
  }

  return Tensor("", out_dtype, {rows, cols}, std::move(out_data));
}

void EyeLike::operator()(const Tensor &input, int64_t k, int32_t dtype, Tensor &output) const {
  Tensor produced = (*this)(input, k, dtype);
  EXT_ENFORCE_INVALID(output.data_type == produced.data_type,
                      "kernel::EyeLike preallocated output must have the expected dtype.");
  EXT_ENFORCE_INVALID(output.shape == produced.shape,
                      "kernel::EyeLike preallocated output shape must match the produced tensor "
                      "shape.");
  EXT_ENFORCE_INVALID(output.data.size() == produced.data.size(),
                      "kernel::EyeLike preallocated output buffer has unexpected size in bytes.");
  if (!produced.data.empty()) {
    std::memcpy(output.data.data(), produced.data.data(), produced.data.size());
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
