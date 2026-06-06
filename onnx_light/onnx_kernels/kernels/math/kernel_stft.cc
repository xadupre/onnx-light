// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kSTFTName = "kernel::STFT";

int64_t ReadInt32OrInt64Scalar(const Tensor &t, const char *field) {
  EXT_ENFORCE_INVALID(t.element_count() == 1,
                      std::string(kSTFTName) + ": " + field + " must be a 0-D tensor.");
  switch (t.data_type) {
  case DataType::INT32:
    return static_cast<int64_t>(t.AsInt32()[0]);
  case DataType::INT64:
    return t.AsInt64()[0];
  default:
    throw std::invalid_argument(std::string(kSTFTName) + ": " + field + " must be INT32 or INT64.");
  }
}

template <typename T>
void StftCompute(const T *signal, const T *window, T *out, int64_t batch_size,
                 int64_t signal_length, int64_t in_last, int64_t n_frames, int64_t frame_step,
                 int64_t frame_length, int64_t dft_unique_bins, bool onesided) {
  (void)onesided;
  const double two_pi = 2.0 * 3.14159265358979323846;
  // Layout of input signal: [batch_size, signal_length, in_last].
  const int64_t in_batch_stride = signal_length * in_last;
  // Layout of output: [batch_size, n_frames, dft_unique_bins, 2].
  const int64_t out_bin_stride = 2;
  const int64_t out_frame_stride = dft_unique_bins * out_bin_stride;
  const int64_t out_batch_stride = n_frames * out_frame_stride;

  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t f = 0; f < n_frames; ++f) {
      const int64_t start = f * frame_step;
      for (int64_t k = 0; k < dft_unique_bins; ++k) {
        double acc_re = 0.0, acc_im = 0.0;
        for (int64_t n = 0; n < frame_length; ++n) {
          const int64_t sample_idx = start + n;
          double xr = 0.0, xi = 0.0;
          if (sample_idx >= 0 && sample_idx < signal_length) {
            const int64_t off = b * in_batch_stride + sample_idx * in_last;
            xr = static_cast<double>(signal[off]);
            if (in_last == 2) {
              xi = static_cast<double>(signal[off + 1]);
            }
          }
          if (window != nullptr) {
            const double w = static_cast<double>(window[n]);
            xr *= w;
            xi *= w;
          }
          const double theta = -two_pi * static_cast<double>(k) * static_cast<double>(n) /
                               static_cast<double>(frame_length);
          const double c = std::cos(theta);
          const double s = std::sin(theta);
          // (xr + i*xi) * (c + i*s) = (xr*c - xi*s) + i*(xr*s + xi*c)
          acc_re += xr * c - xi * s;
          acc_im += xr * s + xi * c;
        }
        const int64_t base = b * out_batch_stride + f * out_frame_stride + k * out_bin_stride;
        out[base] = static_cast<T>(acc_re);
        out[base + 1] = static_cast<T>(acc_im);
      }
    }
  }
}

} // namespace

Tensor STFT::operator()(const Tensor &signal, const Tensor &frame_step, const Tensor *window,
                        const Tensor *frame_length, bool onesided) const {
  const int64_t rank = static_cast<int64_t>(signal.shape.size());
  EXT_ENFORCE_INVALID(rank == 3, std::string(kSTFTName) +
                                     ": signal must have rank 3 ([batch_size, signal_length, 1]"
                                     " or [batch_size, signal_length, 2]).");
  const int64_t batch_size = signal.shape[0];
  const int64_t signal_length = signal.shape[1];
  const int64_t in_last = signal.shape[2];
  EXT_ENFORCE_INVALID(in_last == 1 || in_last == 2,
                      std::string(kSTFTName) +
                          ": the last dimension of signal must be 1 (real) or 2 (complex).");
  if (in_last == 2) {
    EXT_ENFORCE_INVALID(!onesided, std::string(kSTFTName) +
                                       ": onesided is not supported for complex-valued input.");
  }

  const int64_t frame_step_value = ReadInt32OrInt64Scalar(frame_step, "frame_step");
  EXT_ENFORCE_INVALID(frame_step_value > 0,
                      std::string(kSTFTName) + ": frame_step must be positive.");

  // Determine frame_length from inputs.
  int64_t frame_length_value = -1;
  if (frame_length != nullptr) {
    frame_length_value = ReadInt32OrInt64Scalar(*frame_length, "frame_length");
    EXT_ENFORCE_INVALID(frame_length_value > 0,
                        std::string(kSTFTName) + ": frame_length must be positive.");
  }
  if (window != nullptr) {
    EXT_ENFORCE_INVALID(window->shape.size() == 1,
                        std::string(kSTFTName) + ": window must be 1-D.");
    const int64_t window_length = window->shape[0];
    if (frame_length_value < 0) {
      frame_length_value = window_length;
    } else {
      EXT_ENFORCE_INVALID(window_length == frame_length_value,
                          std::string(kSTFTName) +
                              ": window length must match frame_length when both are given.");
    }
  }
  EXT_ENFORCE_INVALID(frame_length_value > 0,
                      std::string(kSTFTName) +
                          ": at least one of window or frame_length must be provided.");
  EXT_ENFORCE_INVALID(frame_length_value <= signal_length,
                      std::string(kSTFTName) + ": frame_length must not exceed signal_length.");

  const int64_t dft_unique_bins = onesided ? ((frame_length_value / 2) + 1) : frame_length_value;
  const int64_t n_frames = (signal_length - frame_length_value) / frame_step_value + 1;

  std::vector<int64_t> out_shape = {batch_size, n_frames, dft_unique_bins, 2};
  int64_t out_total = batch_size * n_frames * dft_unique_bins * 2;
  Tensor output(
      "", signal.data_type, out_shape,
      std::vector<uint8_t>(static_cast<std::size_t>(out_total) * ElementSize(signal.data_type)));

  if (window != nullptr) {
    EXT_ENFORCE_INVALID(window->data_type == signal.data_type,
                        std::string(kSTFTName) + ": window dtype must match signal dtype.");
  }

  switch (signal.data_type) {
  case DataType::FLOAT:
    StftCompute<float>(signal.AsFloat(), window != nullptr ? window->AsFloat() : nullptr,
                       output.AsFloat(), batch_size, signal_length, in_last, n_frames,
                       frame_step_value, frame_length_value, dft_unique_bins, onesided);
    break;
  case DataType::DOUBLE:
    StftCompute<double>(signal.AsDouble(), window != nullptr ? window->AsDouble() : nullptr,
                        output.AsDouble(), batch_size, signal_length, in_last, n_frames,
                        frame_step_value, frame_length_value, dft_unique_bins, onesided);
    break;
  default:
    throw std::invalid_argument(std::string(kSTFTName) +
                                " only supports FLOAT and DOUBLE signal tensors.");
  }
  return output;
}

void STFT::operator()(const Tensor &signal, const Tensor &frame_step, const Tensor *window,
                      const Tensor *frame_length, bool onesided, Tensor &output) const {
  Tensor produced = (*this)(signal, frame_step, window, frame_length, onesided);
  EXT_ENFORCE_INVALID(output.data_type == produced.data_type,
                      std::string(kSTFTName) + ": preallocated output dtype must match.");
  EXT_ENFORCE_INVALID(output.shape == produced.shape,
                      std::string(kSTFTName) + ": preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(output.data.size() == produced.data.size(),
                      std::string(kSTFTName) + ": preallocated output buffer size mismatch.");
  if (!produced.data.empty()) {
    std::memcpy(output.data.data(), produced.data.data(), produced.data.size());
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
