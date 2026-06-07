// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/elementwise_helpers.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {
constexpr const char *kMinName = "kernel::Min";

constexpr const char *kSupportedMinTypesMsg =
    " only supports FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, "
    "UINT32 and UINT64 inputs.";

std::vector<int64_t> BroadcastShape(const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
  const size_t rank = a.size() > b.size() ? a.size() : b.size();
  std::vector<int64_t> sa(rank, 1), sb(rank, 1), out(rank, 1);
  for (size_t i = 0; i < a.size(); ++i) {
    sa[rank - a.size() + i] = a[i];
  }
  for (size_t i = 0; i < b.size(); ++i) {
    sb[rank - b.size() + i] = b[i];
  }
  for (size_t d = 0; d < rank; ++d) {
    if (sa[d] == sb[d] || sa[d] == 1 || sb[d] == 1) {
      out[d] = sa[d] >= sb[d] ? sa[d] : sb[d];
    } else {
      throw std::invalid_argument(std::string(kMinName) +
                                  " input shapes are not multidirectional-broadcastable.");
    }
  }
  return out;
}

std::vector<int64_t> ValidateAndBroadcastShape(const std::vector<Tensor> &inputs,
                                               const char *dtype_name, int32_t expected_dtype) {
  EXT_ENFORCE_INVALID(!inputs.empty(), std::string(kMinName) + " requires at least one input.");
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXT_ENFORCE_INVALID(inputs[i].data_type == expected_dtype,
                        std::string(kMinName) + " only supports " + dtype_name + " tensors.");
  }
  std::vector<int64_t> shape = inputs[0].shape;
  for (size_t i = 1; i < inputs.size(); ++i) {
    shape = BroadcastShape(shape, inputs[i].shape);
  }
  return shape;
}

template <typename T> T MinOf(T a, T b) { return a < b ? a : b; }

template <typename T>
Tensor MinAlloc(const char *dtype_name, int32_t dtype, const std::vector<Tensor> &inputs) {
  const std::vector<int64_t> out_shape = ValidateAndBroadcastShape(inputs, dtype_name, dtype);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  Tensor z("", dtype, out_shape, std::vector<uint8_t>(static_cast<size_t>(out_count) * sizeof(T)));
  if (inputs.size() == 1) {
    std::memcpy(z.data.data(), inputs[0].bytes(),
                static_cast<size_t>(inputs[0].element_count()) * sizeof(T));
    return z;
  }
  detail::BinaryElementwise<T, T>(kMinName, dtype_name, dtype, inputs[0], inputs[1], z, MinOf<T>);
  for (size_t i = 2; i < inputs.size(); ++i) {
    Tensor partial = z;
    detail::BinaryElementwise<T, T>(kMinName, dtype_name, dtype, partial, inputs[i], z, MinOf<T>);
  }
  return z;
}

template <typename T>
void MinInPlace(const char *dtype_name, int32_t dtype, const std::vector<Tensor> &inputs,
                Tensor &output) {
  const std::vector<int64_t> out_shape = ValidateAndBroadcastShape(inputs, dtype_name, dtype);
  const size_t expected_bytes = [&]() {
    int64_t n = 1;
    for (int64_t d : out_shape) {
      n *= d;
    }
    return static_cast<size_t>(n) * sizeof(T);
  }();
  detail::CheckPreallocatedOutput(kMinName, dtype_name, dtype, out_shape, expected_bytes, output);
  if (inputs.size() == 1) {
    std::memcpy(output.data.data(), inputs[0].bytes(),
                static_cast<size_t>(inputs[0].element_count()) * sizeof(T));
    return;
  }
  detail::BinaryElementwise<T, T>(kMinName, dtype_name, dtype, inputs[0], inputs[1], output,
                                  MinOf<T>);
  for (size_t i = 2; i < inputs.size(); ++i) {
    Tensor partial = output;
    detail::BinaryElementwise<T, T>(kMinName, dtype_name, dtype, partial, inputs[i], output,
                                    MinOf<T>);
  }
}

#define ONNX_LIGHT_MIN_DISPATCH(MACRO)                                                             \
  MACRO(FLOAT, float, "FLOAT")                                                                     \
  MACRO(DOUBLE, double, "DOUBLE")                                                                  \
  MACRO(INT8, int8_t, "INT8")                                                                      \
  MACRO(INT16, int16_t, "INT16")                                                                   \
  MACRO(INT32, int32_t, "INT32")                                                                   \
  MACRO(INT64, int64_t, "INT64")                                                                   \
  MACRO(UINT8, uint8_t, "UINT8")                                                                   \
  MACRO(UINT16, uint16_t, "UINT16")                                                                \
  MACRO(UINT32, uint32_t, "UINT32")                                                                \
  MACRO(UINT64, uint64_t, "UINT64")

} // namespace

Tensor Min::operator()(const std::vector<Tensor> &inputs) const {
  EXT_ENFORCE_INVALID(!inputs.empty(), std::string(kMinName) + " requires at least one input.");
  switch (inputs[0].data_type) {
#define ONNX_LIGHT_MIN_CASE_ALLOC(ENUM, CPP, NAME)                                                 \
  case DataType::ENUM:                                                                             \
    return MinAlloc<CPP>(NAME, DataType::ENUM, inputs);
    ONNX_LIGHT_MIN_DISPATCH(ONNX_LIGHT_MIN_CASE_ALLOC)
#undef ONNX_LIGHT_MIN_CASE_ALLOC
  default:
    throw std::invalid_argument(std::string(kMinName) + kSupportedMinTypesMsg);
  }
}

void Min::operator()(const std::vector<Tensor> &inputs, Tensor &output) const {
  EXT_ENFORCE_INVALID(!inputs.empty(), std::string(kMinName) + " requires at least one input.");
  switch (inputs[0].data_type) {
#define ONNX_LIGHT_MIN_CASE_INPLACE(ENUM, CPP, NAME)                                               \
  case DataType::ENUM:                                                                             \
    return MinInPlace<CPP>(NAME, DataType::ENUM, inputs, output);
    ONNX_LIGHT_MIN_DISPATCH(ONNX_LIGHT_MIN_CASE_INPLACE)
#undef ONNX_LIGHT_MIN_CASE_INPLACE
  default:
    throw std::invalid_argument(std::string(kMinName) + kSupportedMinTypesMsg);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
