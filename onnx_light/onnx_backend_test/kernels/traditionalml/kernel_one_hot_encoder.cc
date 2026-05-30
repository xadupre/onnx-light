// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

template <typename T> int32_t InputDataType() noexcept { return TensorElementType<T>::value; }

template <typename T> int64_t AsInt64(const T &value) {
  if constexpr (std::is_same_v<T, std::string>) {
    return 0; // Unused; string inputs go through the string overload.
  } else {
    return static_cast<int64_t>(value);
  }
}

template <typename T> void ValidateNumericInput(const Tensor &x, const std::vector<int64_t> &cats) {
  EXT_ENFORCE_INVALID(x.data_type == InputDataType<T>(),
                      "kernel::OneHotEncoder input data_type does not match the requested T.");
  EXT_ENFORCE_INVALID(!cats.empty(),
                      "kernel::OneHotEncoder requires at least one category in cats_int64s.");
}

void ValidateStringInput(const Tensor &x, const std::vector<std::string> &cats) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(TensorProto::DataType::STRING),
                      "kernel::OneHotEncoder expects a STRING input for string categories.");
  EXT_ENFORCE_INVALID(static_cast<int64_t>(x.string_data.size()) == x.element_count(),
                      "kernel::OneHotEncoder STRING input string_data size does not match shape.");
  EXT_ENFORCE_INVALID(!cats.empty(),
                      "kernel::OneHotEncoder requires at least one category in cats_strings.");
}

std::vector<int64_t> OneHotShape(const std::vector<int64_t> &input_shape, int64_t num_cats) {
  std::vector<int64_t> out_shape = input_shape;
  out_shape.push_back(num_cats);
  return out_shape;
}

template <typename T>
void FillOneHotNumeric(const Tensor &x, const std::vector<int64_t> &cats, bool zeros, float *out) {
  const int64_t n = x.element_count();
  const int64_t k = static_cast<int64_t>(cats.size());
  const T *px = x.As<T>();
  std::memset(out, 0, static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(float));
  for (int64_t i = 0; i < n; ++i) {
    const int64_t value = AsInt64<T>(px[i]);
    bool matched = false;
    for (int64_t j = 0; j < k; ++j) {
      if (cats[static_cast<size_t>(j)] == value) {
        out[i * k + j] = 1.0f;
        matched = true;
        break;
      }
    }
    if (!matched && !zeros) {
      throw std::invalid_argument(
          "kernel::OneHotEncoder: input value not found in cats_int64s and zeros=false.");
    }
  }
}

void FillOneHotString(const Tensor &x, const std::vector<std::string> &cats, bool zeros,
                      float *out) {
  const int64_t n = x.element_count();
  const int64_t k = static_cast<int64_t>(cats.size());
  const std::vector<std::string> &px = x.AsStrings();
  std::memset(out, 0, static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(float));
  for (int64_t i = 0; i < n; ++i) {
    bool matched = false;
    for (int64_t j = 0; j < k; ++j) {
      if (px[static_cast<size_t>(i)] == cats[static_cast<size_t>(j)]) {
        out[i * k + j] = 1.0f;
        matched = true;
        break;
      }
    }
    if (!matched && !zeros) {
      throw std::invalid_argument(
          "kernel::OneHotEncoder: input value not found in cats_strings and zeros=false.");
    }
  }
}

void ValidatePreallocatedOutput(const Tensor &output, const std::vector<int64_t> &expected_shape) {
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorProto::DataType::FLOAT),
                      "kernel::OneHotEncoder preallocated output dtype must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::OneHotEncoder preallocated output shape does not match the expected "
                      "one-hot output shape.");
  int64_t expected_n = 1;
  for (int64_t d : expected_shape) {
    expected_n *= d;
  }
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(expected_n) * sizeof(float),
                      "kernel::OneHotEncoder preallocated output buffer is incorrectly sized.");
}

} // namespace

template <typename T>
Tensor OneHotEncoder::operator()(const Tensor &x, const std::vector<int64_t> &cats,
                                 bool zeros) const {
  ValidateNumericInput<T>(x, cats);
  const std::vector<int64_t> out_shape = OneHotShape(x.shape, static_cast<int64_t>(cats.size()));
  const int64_t total = x.element_count() * static_cast<int64_t>(cats.size());
  std::vector<uint8_t> bytes(static_cast<size_t>(total) * sizeof(float));
  Tensor out("", static_cast<int32_t>(TensorProto::DataType::FLOAT), out_shape, std::move(bytes));
  FillOneHotNumeric<T>(x, cats, zeros, reinterpret_cast<float *>(out.data.data()));
  return out;
}

Tensor OneHotEncoder::operator()(const Tensor &x, const std::vector<std::string> &cats,
                                 bool zeros) const {
  ValidateStringInput(x, cats);
  const std::vector<int64_t> out_shape = OneHotShape(x.shape, static_cast<int64_t>(cats.size()));
  const int64_t total = x.element_count() * static_cast<int64_t>(cats.size());
  std::vector<uint8_t> bytes(static_cast<size_t>(total) * sizeof(float));
  Tensor out("", static_cast<int32_t>(TensorProto::DataType::FLOAT), out_shape, std::move(bytes));
  FillOneHotString(x, cats, zeros, reinterpret_cast<float *>(out.data.data()));
  return out;
}

template <typename T>
void OneHotEncoder::operator()(const Tensor &x, const std::vector<int64_t> &cats, bool zeros,
                               Tensor &output) const {
  ValidateNumericInput<T>(x, cats);
  const std::vector<int64_t> expected_shape =
      OneHotShape(x.shape, static_cast<int64_t>(cats.size()));
  ValidatePreallocatedOutput(output, expected_shape);
  FillOneHotNumeric<T>(x, cats, zeros, output.AsFloat());
}

void OneHotEncoder::operator()(const Tensor &x, const std::vector<std::string> &cats, bool zeros,
                               Tensor &output) const {
  ValidateStringInput(x, cats);
  const std::vector<int64_t> expected_shape =
      OneHotShape(x.shape, static_cast<int64_t>(cats.size()));
  ValidatePreallocatedOutput(output, expected_shape);
  FillOneHotString(x, cats, zeros, output.AsFloat());
}

// Explicit instantiations for the supported numeric element types.
#define ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER(T)                                                  \
  template Tensor OneHotEncoder::operator()<T>(const Tensor &, const std::vector<int64_t> &, bool) \
      const;                                                                                       \
  template void OneHotEncoder::operator()<T>(const Tensor &, const std::vector<int64_t> &, bool,   \
                                             Tensor &) const

ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER(int64_t);
ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER(int32_t);
ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER(float);
ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER(double);

#undef ONNX_LIGHT_INSTANTIATE_ONE_HOT_ENCODER

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
