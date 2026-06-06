// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

int64_t ReadIntScalar(const Tensor &t, const char *what) {
  EXT_ENFORCE_INVALID(t.element_count() == 1 && t.shape.empty(),
                      "kernel::MelWeightMatrix expects a scalar tensor.");
  if (t.data_type == DataType::INT32) {
    return static_cast<int64_t>(t.AsInt32()[0]);
  }
  if (t.data_type == DataType::INT64) {
    return t.AsInt64()[0];
  }
  EXT_THROW("kernel::MelWeightMatrix ", what, " must be an INT32 or INT64 tensor.");
}

double ReadFloatScalar(const Tensor &t, const char *what) {
  EXT_ENFORCE_INVALID(t.element_count() == 1 && t.shape.empty(),
                      "kernel::MelWeightMatrix expects a scalar tensor.");
  if (t.data_type == DataType::FLOAT) {
    return static_cast<double>(t.AsFloat()[0]);
  }
  if (t.data_type == DataType::DOUBLE) {
    return t.AsDouble()[0];
  }
  EXT_THROW("kernel::MelWeightMatrix ", what, " must be a FLOAT or DOUBLE tensor.");
}

template <typename T> void FillOutput(const std::vector<double> &values, Tensor &output) {
  T *po = reinterpret_cast<T *>(output.data.data());
  for (size_t i = 0; i < values.size(); ++i) {
    po[i] = static_cast<T>(values[i]);
  }
}

} // namespace

Tensor MelWeightMatrix::operator()(const Tensor &num_mel_bins, const Tensor &dft_length,
                                   const Tensor &sample_rate, const Tensor &lower_edge_hertz,
                                   const Tensor &upper_edge_hertz, DataType output_dtype) const {
  const int64_t num_mel_bins_v = ReadIntScalar(num_mel_bins, "num_mel_bins");
  const int64_t dft_length_v = ReadIntScalar(dft_length, "dft_length");
  EXT_ENFORCE_INVALID(num_mel_bins_v > 0, "kernel::MelWeightMatrix num_mel_bins must be positive.");
  EXT_ENFORCE_INVALID(dft_length_v > 0, "kernel::MelWeightMatrix dft_length must be positive.");

  const int64_t num_spectrogram_bins = dft_length_v / 2 + 1;
  size_t element_bytes;
  switch (output_dtype) {
  case DataType::FLOAT:
    element_bytes = sizeof(float);
    break;
  case DataType::DOUBLE:
    element_bytes = sizeof(double);
    break;
  default:
    EXT_THROW("kernel::MelWeightMatrix output_dtype must be FLOAT or DOUBLE.");
  }
  const size_t total = static_cast<size_t>(num_spectrogram_bins * num_mel_bins_v);
  Tensor y("", output_dtype, {num_spectrogram_bins, num_mel_bins_v},
           std::vector<uint8_t>(total * element_bytes, 0));
  (*this)(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz, output_dtype,
          y);
  return y;
}

void MelWeightMatrix::operator()(const Tensor &num_mel_bins, const Tensor &dft_length,
                                 const Tensor &sample_rate, const Tensor &lower_edge_hertz,
                                 const Tensor &upper_edge_hertz, DataType output_dtype,
                                 Tensor &output) const {
  const int64_t num_mel_bins_v = ReadIntScalar(num_mel_bins, "num_mel_bins");
  const int64_t dft_length_v = ReadIntScalar(dft_length, "dft_length");
  const int64_t sample_rate_v = ReadIntScalar(sample_rate, "sample_rate");
  const double lower_hz = ReadFloatScalar(lower_edge_hertz, "lower_edge_hertz");
  const double upper_hz = ReadFloatScalar(upper_edge_hertz, "upper_edge_hertz");

  EXT_ENFORCE_INVALID(num_mel_bins_v > 0, "kernel::MelWeightMatrix num_mel_bins must be positive.");
  EXT_ENFORCE_INVALID(dft_length_v > 0, "kernel::MelWeightMatrix dft_length must be positive.");
  EXT_ENFORCE_INVALID(sample_rate_v > 0, "kernel::MelWeightMatrix sample_rate must be positive.");
  EXT_ENFORCE_INVALID(output.data_type == output_dtype,
                      "kernel::MelWeightMatrix preallocated output data_type mismatch.");

  const int64_t num_spectrogram_bins = dft_length_v / 2 + 1;
  EXT_ENFORCE_INVALID(output.shape.size() == 2 && output.shape[0] == num_spectrogram_bins &&
                          output.shape[1] == num_mel_bins_v,
                      "kernel::MelWeightMatrix preallocated output shape must be "
                      "{floor(dft_length/2) + 1, num_mel_bins}.");

  // Reference implementation: see
  // https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_mel_weight_matrix.py
  const int64_t n_points = num_mel_bins_v + 2;
  const double low_mel = 2595.0 * std::log10(1.0 + lower_hz / 700.0);
  const double high_mel = 2595.0 * std::log10(1.0 + upper_hz / 700.0);
  const double mel_step = (high_mel - low_mel) / static_cast<double>(n_points);

  std::vector<int64_t> bin_indices(static_cast<size_t>(n_points));
  for (int64_t i = 0; i < n_points; ++i) {
    const double mel_value = static_cast<double>(i) * mel_step + low_mel;
    const double hz = 700.0 * (std::pow(10.0, mel_value / 2595.0) - 1.0);
    // floor division to match NumPy's ``//`` on non-negative values.
    const double scaled =
        (static_cast<double>(dft_length_v) + 1.0) * hz / static_cast<double>(sample_rate_v);
    bin_indices[static_cast<size_t>(i)] = static_cast<int64_t>(std::floor(scaled));
  }

  const size_t total = static_cast<size_t>(num_spectrogram_bins * num_mel_bins_v);
  std::vector<double> values(total, 0.0);
  for (int64_t i = 0; i < num_mel_bins_v; ++i) {
    const int64_t lower = bin_indices[static_cast<size_t>(i)];
    const int64_t center = bin_indices[static_cast<size_t>(i + 1)];
    const int64_t higher = bin_indices[static_cast<size_t>(i + 2)];

    const int64_t low_to_center = center - lower;
    if (low_to_center == 0) {
      if (center >= 0 && center < num_spectrogram_bins) {
        values[static_cast<size_t>(center * num_mel_bins_v + i)] = 1.0;
      }
    } else {
      for (int64_t j = lower; j <= center; ++j) {
        if (j < 0 || j >= num_spectrogram_bins) {
          continue;
        }
        values[static_cast<size_t>(j * num_mel_bins_v + i)] =
            static_cast<double>(j - lower) / static_cast<double>(low_to_center);
      }
    }
    const int64_t center_to_high = higher - center;
    if (center_to_high > 0) {
      for (int64_t j = center; j < higher; ++j) {
        if (j < 0 || j >= num_spectrogram_bins) {
          continue;
        }
        values[static_cast<size_t>(j * num_mel_bins_v + i)] =
            static_cast<double>(higher - j) / static_cast<double>(center_to_high);
      }
    }
  }

  if (output_dtype == DataType::FLOAT) {
    FillOutput<float>(values, output);
  } else if (output_dtype == DataType::DOUBLE) {
    FillOutput<double>(values, output);
  } else {
    EXT_THROW("kernel::MelWeightMatrix output_dtype must be FLOAT or DOUBLE.");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
