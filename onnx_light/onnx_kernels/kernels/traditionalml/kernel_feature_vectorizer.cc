// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Decomposes a tensor's logical shape into (batch, feature) dims. A rank-1
// tensor [C] is treated as [1, C]. A rank-2 tensor [N, C] is used as-is.
std::pair<int64_t, int64_t> BatchFeature(const Tensor &t) {
  EXT_ENFORCE_INVALID(t.shape.size() == 1 || t.shape.size() == 2,
                      "kernel::FeatureVectorizer: each input must be rank 1 or 2.");
  if (t.shape.size() == 1) {
    return {1, t.shape[0]};
  }
  return {t.shape[0], t.shape[1]};
}

template <typename T> float ToFloat(T v) { return static_cast<float>(v); }

template <typename T>
void CopyRowAsFloat(const Tensor &t, int64_t row, int64_t declared_features, float *out_row,
                    int64_t out_feature_offset) {
  const T *p = t.As<T>();
  const auto [batch, features] = BatchFeature(t);
  // Inputs whose batch dim is smaller than the global batch (i.e., 1) broadcast
  // row 0 to every output row. This is the FeatureVectorizer convention used by
  // ONNX runtime when only one input has [1,F].
  const int64_t src_row = (row < batch) ? row : 0;
  const T *src = p + src_row * features;
  const int64_t copy = (declared_features < features) ? declared_features : features;
  for (int64_t i = 0; i < copy; ++i) {
    out_row[out_feature_offset + i] = ToFloat(src[i]);
  }
  // Zero-pad if the declared feature width is wider than the actual input.
  for (int64_t i = copy; i < declared_features; ++i) {
    out_row[out_feature_offset + i] = 0.0f;
  }
}

void DispatchCopyRowAsFloat(const Tensor &t, int64_t row, int64_t declared_features, float *out_row,
                            int64_t out_feature_offset) {
  switch (static_cast<DataType>(t.data_type)) {
  case DataType::FLOAT:
    CopyRowAsFloat<float>(t, row, declared_features, out_row, out_feature_offset);
    return;
  case DataType::DOUBLE:
    CopyRowAsFloat<double>(t, row, declared_features, out_row, out_feature_offset);
    return;
  case DataType::INT32:
    CopyRowAsFloat<int32_t>(t, row, declared_features, out_row, out_feature_offset);
    return;
  case DataType::INT64:
    CopyRowAsFloat<int64_t>(t, row, declared_features, out_row, out_feature_offset);
    return;
  default:
    throw std::invalid_argument("kernel::FeatureVectorizer: input element type must be one of "
                                "float/double/int32/int64.");
  }
}

void ValidateInputs(const std::vector<Tensor> &inputs,
                    const std::vector<int64_t> &inputdimensions) {
  EXT_ENFORCE_INVALID(!inputs.empty(), "kernel::FeatureVectorizer requires at least one input.");
  if (!inputdimensions.empty()) {
    EXT_ENFORCE_INVALID(inputdimensions.size() == inputs.size(),
                        "kernel::FeatureVectorizer: 'inputdimensions' length must match the "
                        "number of inputs when provided.");
    for (int64_t d : inputdimensions) {
      EXT_ENFORCE_INVALID(d > 0, "kernel::FeatureVectorizer: 'inputdimensions' entries must be "
                                 "strictly positive.");
    }
  }
  for (const Tensor &t : inputs) {
    (void)BatchFeature(t); // validates rank
  }
}

std::vector<int64_t> ResolveInputDims(const std::vector<Tensor> &inputs,
                                      const std::vector<int64_t> &inputdimensions) {
  if (!inputdimensions.empty()) {
    return inputdimensions;
  }
  std::vector<int64_t> dims;
  dims.reserve(inputs.size());
  for (const Tensor &t : inputs) {
    dims.push_back(BatchFeature(t).second);
  }
  return dims;
}

int64_t ResolveBatchSize(const std::vector<Tensor> &inputs) {
  int64_t batch = 1;
  for (const Tensor &t : inputs) {
    const int64_t b = BatchFeature(t).first;
    if (b > batch) {
      batch = b;
    }
  }
  return batch;
}

void Compute(const std::vector<Tensor> &inputs, const std::vector<int64_t> &input_dims, int64_t n,
             int64_t total_features, float *out) {
  for (int64_t row = 0; row < n; ++row) {
    float *out_row = out + row * total_features;
    int64_t offset = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      DispatchCopyRowAsFloat(inputs[i], row, input_dims[i], out_row, offset);
      offset += input_dims[i];
    }
  }
}

} // namespace

Tensor FeatureVectorizer::operator()(const std::vector<Tensor> &inputs,
                                     const std::vector<int64_t> &inputdimensions) const {
  ValidateInputs(inputs, inputdimensions);
  const std::vector<int64_t> input_dims = ResolveInputDims(inputs, inputdimensions);
  const int64_t n = ResolveBatchSize(inputs);
  int64_t total_features = 0;
  for (int64_t d : input_dims) {
    total_features += d;
  }
  std::vector<float> values(static_cast<size_t>(n * total_features), 0.0f);
  Compute(inputs, input_dims, n, total_features, values.data());
  return Tensor::FromFloat("", {n, total_features}, values);
}

void FeatureVectorizer::operator()(const std::vector<Tensor> &inputs,
                                   const std::vector<int64_t> &inputdimensions,
                                   Tensor &output) const {
  ValidateInputs(inputs, inputdimensions);
  const std::vector<int64_t> input_dims = ResolveInputDims(inputs, inputdimensions);
  const int64_t n = ResolveBatchSize(inputs);
  int64_t total_features = 0;
  for (int64_t d : input_dims) {
    total_features += d;
  }
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::FeatureVectorizer preallocated output dtype must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == (std::vector<int64_t>{n, total_features}),
                      "kernel::FeatureVectorizer preallocated output shape must be "
                      "[batch, sum(input_dims)].");
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(n * total_features) * sizeof(float),
                      "kernel::FeatureVectorizer preallocated output buffer is incorrectly sized.");
  std::fill(output.data.begin(), output.data.end(), uint8_t{0u});
  Compute(inputs, input_dims, n, total_features, reinterpret_cast<float *>(output.data.data()));
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
