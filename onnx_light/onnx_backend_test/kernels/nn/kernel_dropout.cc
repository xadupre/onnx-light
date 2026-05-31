// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

constexpr uint32_t kDefaultDropoutSeed = 0u;

void ValidateInput(const Tensor &data, float ratio) {
  EXT_ENFORCE_INVALID(ratio >= 0.0f && ratio < 1.0f, "kernel::Dropout: ratio must be in [0, 1).");
  EXT_ENFORCE_INVALID(data.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          data.data_type == static_cast<int32_t>(DataType::DOUBLE),
                      "kernel::Dropout: only FLOAT and DOUBLE are supported.");
}

template <typename T>
void ComputeDropout(const T *src, T *dst, uint8_t *mask_data, int64_t n, float ratio,
                    bool training_mode, uint32_t seed) {
  if (!training_mode || ratio == 0.0f) {
    if (dst != src) {
      std::memcpy(dst, src, static_cast<std::size_t>(n) * sizeof(T));
    }
    if (mask_data != nullptr) {
      std::fill(mask_data, mask_data + n, static_cast<uint8_t>(1));
    }
    return;
  }

  const float scale = 1.0f / (1.0f - ratio);
  std::mt19937 engine(seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for (int64_t i = 0; i < n; ++i) {
    const bool keep = uniform(engine) >= ratio;
    if (mask_data != nullptr) {
      mask_data[i] = keep ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    }
    dst[i] = keep ? static_cast<T>(src[i] * static_cast<T>(scale)) : static_cast<T>(0);
  }
}

} // namespace

std::pair<Tensor, Tensor> Dropout::operator()(const Tensor &data, float ratio, bool training_mode,
                                              int64_t seed) const {
  ValidateInput(data, ratio);

  Tensor output("", data.data_type, data.shape, std::vector<uint8_t>(data.data.size()));
  Tensor mask("", static_cast<int32_t>(DataType::BOOL), data.shape,
              std::vector<uint8_t>(static_cast<std::size_t>(data.element_count()), 1));

  output = (*this)(data, ratio, training_mode, mask, seed);
  return {std::move(output), std::move(mask)};
}

Tensor Dropout::operator()(const Tensor &data, float ratio, bool training_mode, Tensor &mask,
                           int64_t seed) const {
  ValidateInput(data, ratio);
  EXT_ENFORCE_INVALID(mask.data_type == static_cast<int32_t>(DataType::BOOL),
                      "kernel::Dropout: mask must have BOOL dtype.");
  EXT_ENFORCE_INVALID(mask.shape == data.shape, "kernel::Dropout: mask shape must match input.");
  EXT_ENFORCE_INVALID(mask.data.size() == static_cast<std::size_t>(data.element_count()),
                      "kernel::Dropout: mask buffer must have one byte per input element.");

  Tensor output("", data.data_type, data.shape, std::vector<uint8_t>(data.data.size()));
  const uint32_t engine_seed =
      (seed == kNoSeed) ? kDefaultDropoutSeed : static_cast<uint32_t>(seed);

  const int64_t n = data.element_count();
  if (data.data_type == static_cast<int32_t>(DataType::FLOAT)) {
    ComputeDropout<float>(data.AsFloat(), output.AsFloat(), mask.AsBool(), n, ratio, training_mode,
                          engine_seed);
  } else {
    ComputeDropout<double>(data.AsDouble(), output.AsDouble(), mask.AsBool(), n, ratio,
                           training_mode, engine_seed);
  }
  return output;
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
