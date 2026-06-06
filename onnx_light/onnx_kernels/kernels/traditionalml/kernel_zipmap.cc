// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

template <typename T>
std::vector<int64_t> ValidateAndComputeOutputShape(const Tensor &x,
                                                   const std::vector<T> &class_labels) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::ZipMap expects a float input tensor, got data_type=" +
                          std::to_string(x.data_type) + ".");
  EXT_ENFORCE_INVALID(!class_labels.empty(), "kernel::ZipMap expects non-empty class labels.");
  EXT_ENFORCE_INVALID(x.shape.size() == 1 || x.shape.size() == 2,
                      "kernel::ZipMap expects input rank 1 or 2.");
  for (int64_t d : x.shape) {
    EXT_ENFORCE_INVALID(d >= 0, "kernel::ZipMap input shape dimensions must be non-negative.");
  }

  const int64_t class_count = x.shape.back();
  EXT_ENFORCE_INVALID(class_count == static_cast<int64_t>(class_labels.size()),
                      "kernel::ZipMap class labels size must match input class dimension, got " +
                          std::to_string(class_labels.size()) + " labels for class dimension " +
                          std::to_string(class_count) + ".");

  if (x.shape.size() == 1) {
    return std::vector<int64_t>{1, class_count};
  }
  return x.shape;
}

template <typename T>
Tensor ComputeZipMapOutput(const Tensor &x, const std::vector<T> &class_labels) {
  const std::vector<int64_t> output_shape = ValidateAndComputeOutputShape(x, class_labels);
  std::vector<uint8_t> output_bytes = x.data;
  return Tensor("", static_cast<int32_t>(DataType::FLOAT), output_shape, std::move(output_bytes));
}

template <typename T>
void ComputeZipMapOutput(const Tensor &x, const std::vector<T> &class_labels, Tensor &output) {
  const std::vector<int64_t> expected_shape = ValidateAndComputeOutputShape(x, class_labels);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::ZipMap preallocated output dtype must be float.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::ZipMap preallocated output shape is incorrect.");
  EXT_ENFORCE_INVALID(output.data.size() == x.data.size(),
                      "kernel::ZipMap preallocated output buffer is incorrectly sized.");
  if (!x.data.empty()) {
    std::memcpy(output.data.data(), x.data.data(), x.data.size());
  }
}

} // namespace

Tensor ZipMap::operator()(const Tensor &x, const std::vector<int64_t> &class_labels) const {
  return ComputeZipMapOutput(x, class_labels);
}

Tensor ZipMap::operator()(const Tensor &x, const std::vector<std::string> &class_labels) const {
  return ComputeZipMapOutput(x, class_labels);
}

void ZipMap::operator()(const Tensor &x, const std::vector<int64_t> &class_labels,
                        Tensor &output) const {
  ComputeZipMapOutput(x, class_labels, output);
}

void ZipMap::operator()(const Tensor &x, const std::vector<std::string> &class_labels,
                        Tensor &output) const {
  ComputeZipMapOutput(x, class_labels, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
