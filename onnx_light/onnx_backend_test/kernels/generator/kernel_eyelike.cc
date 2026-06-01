// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
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

  std::vector<uint8_t> one(es, uint8_t{0});
  one[0] = uint8_t{1};
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
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
