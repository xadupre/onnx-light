// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

enum class NormMode { kMax, kL1, kL2 };

NormMode ParseNorm(const std::string &norm) {
  if (norm == "MAX") {
    return NormMode::kMax;
  }
  if (norm == "L1") {
    return NormMode::kL1;
  }
  if (norm == "L2") {
    return NormMode::kL2;
  }
  EXT_ENFORCE_INVALID(false, "kernel::Normalizer 'norm' must be one of 'MAX', 'L1', 'L2'.");
  return NormMode::kMax; // unreachable
}

template <typename T> void ValidateInput(const Tensor &x) {
  EXT_ENFORCE_INVALID(x.data_type == TensorElementType<T>::value,
                      "kernel::Normalizer input data_type does not match the requested element T.");
  EXT_ENFORCE_INVALID(x.shape.size() == 1u || x.shape.size() == 2u,
                      "kernel::Normalizer requires an input of rank 1 ([C]) or rank 2 ([N,C]).");
}

void RowExtent(const std::vector<int64_t> &shape, int64_t &rows, int64_t &cols) {
  if (shape.size() == 1u) {
    rows = 1;
    cols = shape[0];
  } else {
    rows = shape[0];
    cols = shape[1];
  }
}

template <typename T> void ApplyNormalizer(const Tensor &x, NormMode mode, float *out) {
  const T *px = x.As<T>();
  int64_t rows = 0;
  int64_t cols = 0;
  RowExtent(x.shape, rows, cols);
  for (int64_t r = 0; r < rows; ++r) {
    const T *row_in = px + r * cols;
    float *row_out = out + r * cols;
    double divisor = 0.0;
    switch (mode) {
    case NormMode::kMax: {
      // Match ai.onnx.ml::Normalizer / onnxruntime semantics: divisor is the
      // signed maximum value over the row (not max(|x|)).
      double m = std::numeric_limits<double>::lowest();
      for (int64_t c = 0; c < cols; ++c) {
        const double v = static_cast<double>(row_in[c]);
        if (v > m) {
          m = v;
        }
      }
      divisor = m;
      break;
    }
    case NormMode::kL1: {
      double s = 0.0;
      for (int64_t c = 0; c < cols; ++c) {
        s += std::fabs(static_cast<double>(row_in[c]));
      }
      divisor = s;
      break;
    }
    case NormMode::kL2: {
      double s = 0.0;
      for (int64_t c = 0; c < cols; ++c) {
        const double v = static_cast<double>(row_in[c]);
        s += v * v;
      }
      divisor = std::sqrt(s);
      break;
    }
    }
    if (divisor == 0.0) {
      // Per the ONNX spec: if the divisor is zero, Y == X.
      for (int64_t c = 0; c < cols; ++c) {
        row_out[c] = static_cast<float>(row_in[c]);
      }
    } else {
      const double inv = 1.0 / divisor;
      for (int64_t c = 0; c < cols; ++c) {
        row_out[c] = static_cast<float>(static_cast<double>(row_in[c]) * inv);
      }
    }
  }
}

} // namespace

template <typename T>
Tensor Normalizer::operator()(const Tensor &x, const std::string &norm) const {
  ValidateInput<T>(x);
  const NormMode mode = ParseNorm(norm);
  const int64_t n = x.element_count();
  std::vector<uint8_t> bytes(static_cast<size_t>(n) * sizeof(float));
  Tensor out("", TensorElementType<float>::value, x.shape, std::move(bytes));
  ApplyNormalizer<T>(x, mode, reinterpret_cast<float *>(out.data.data()));
  return out;
}

template <typename T>
void Normalizer::operator()(const Tensor &x, const std::string &norm, Tensor &output) const {
  ValidateInput<T>(x);
  const NormMode mode = ParseNorm(norm);
  EXT_ENFORCE_INVALID(output.data_type == TensorElementType<float>::value,
                      "kernel::Normalizer preallocated output dtype must be float.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Normalizer preallocated output shape must match the input shape.");
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(x.element_count()) * sizeof(float),
                      "kernel::Normalizer preallocated output buffer is incorrectly sized.");
  ApplyNormalizer<T>(x, mode, output.As<float>());
}

// Explicit instantiations for the supported element types.
#define ONNX_LIGHT_INSTANTIATE_NORMALIZER(T)                                                       \
  template Tensor Normalizer::operator()<T>(const Tensor &, const std::string &) const;            \
  template void Normalizer::operator()<T>(const Tensor &, const std::string &, Tensor &) const

ONNX_LIGHT_INSTANTIATE_NORMALIZER(float);
ONNX_LIGHT_INSTANTIATE_NORMALIZER(double);
ONNX_LIGHT_INSTANTIATE_NORMALIZER(int64_t);
ONNX_LIGHT_INSTANTIATE_NORMALIZER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_NORMALIZER

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
