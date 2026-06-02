// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

constexpr const char *kDFTName = "kernel::DFT";

int64_t ReadDFTLength(const Tensor &len) {
  EXT_ENFORCE_INVALID(len.element_count() == 1,
                      std::string(kDFTName) + ": dft_length must be a 0-D tensor.");
  switch (len.data_type) {
  case DataType::INT32:
    return static_cast<int64_t>(len.AsInt32()[0]);
  case DataType::INT64:
    return len.AsInt64()[0];
  default:
    throw std::invalid_argument(std::string(kDFTName) + ": dft_length must be INT32 or INT64.");
  }
}

// Read one sample (real, imag) at the given multi-index from a tensor whose
// trailing dimension is 1 (real) or 2 (complex). Returns (0, 0) for indices
// that are out of range along ``axis`` (zero-padding for dft_length > N).
template <typename T>
void ReadSample(const T *data, const std::vector<int64_t> &strides, int64_t axis_dim, int64_t outer,
                int64_t inner, int64_t outer_idx, int64_t axis_idx, int64_t inner_idx,
                int64_t last_dim, double &re, double &im) {
  (void)outer;
  (void)inner;
  if (axis_idx < 0 || axis_idx >= axis_dim) {
    re = 0.0;
    im = 0.0;
    return;
  }
  const int64_t off = outer_idx * strides[0] + axis_idx * strides[1] + inner_idx * strides[2];
  re = static_cast<double>(data[off]);
  im = (last_dim == 2) ? static_cast<double>(data[off + 1]) : 0.0;
}

template <typename T>
void DftCompute(const T *in, T *out, int64_t outer, int64_t in_axis, int64_t out_axis,
                int64_t inner, int64_t in_last, int64_t out_last, int64_t n_dft, bool inverse,
                bool onesided) {
  // Strides for INPUT  (outer, axis, inner_with_last). inner_with_last = inner * in_last.
  const std::vector<int64_t> in_strides = {in_axis * inner * in_last, inner * in_last, in_last};
  // Strides for OUTPUT.
  const int64_t out_inner_stride = out_last;
  const int64_t out_axis_stride = inner * out_inner_stride;
  const int64_t out_outer_stride = out_axis * out_axis_stride;

  const double sign = inverse ? +1.0 : -1.0;
  const double norm = inverse ? (1.0 / static_cast<double>(n_dft)) : 1.0;
  const double two_pi = 2.0 * 3.14159265358979323846;

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      for (int64_t k = 0; k < out_axis; ++k) {
        double acc_re = 0.0, acc_im = 0.0;
        for (int64_t n = 0; n < n_dft; ++n) {
          double xr = 0.0, xi = 0.0;
          ReadSample<T>(in, in_strides, in_axis, outer, inner, o, n, i, in_last, xr, xi);
          const double theta = sign * two_pi * static_cast<double>(k) * static_cast<double>(n) /
                               static_cast<double>(n_dft);
          const double c = std::cos(theta);
          const double s = std::sin(theta);
          // (xr + i*xi) * (c + i*s) = (xr*c - xi*s) + i*(xr*s + xi*c)
          acc_re += xr * c - xi * s;
          acc_im += xr * s + xi * c;
        }
        acc_re *= norm;
        acc_im *= norm;
        const int64_t base = o * out_outer_stride + k * out_axis_stride + i * out_inner_stride;
        if (out_last == 2) {
          out[base] = static_cast<T>(acc_re);
          out[base + 1] = static_cast<T>(acc_im);
        } else {
          // IRFFT: take the real part only.
          (void)onesided;
          out[base] = static_cast<T>(acc_re);
        }
      }
    }
  }
}

} // namespace

Tensor DFT::operator()(const Tensor &input, const Tensor *dft_length, int64_t axis, bool onesided,
                       bool inverse) const {
  const int64_t rank = static_cast<int64_t>(input.shape.size());
  EXT_ENFORCE_INVALID(rank >= 2, std::string(kDFTName) +
                                     ": input must have rank >= 2 (including the "
                                     "trailing complex/real dimension).");
  // Normalise axis (negative -> positive). The valid range matches the schema:
  // ``-rank <= axis``, ``axis != -1`` and ``axis < rank - 1``.
  EXT_ENFORCE_INVALID(axis >= -rank && axis != -1 && axis < rank - 1,
                      std::string(kDFTName) + ": axis is out of range.");
  const int64_t a = axis < 0 ? axis + rank : axis;
  const int64_t last_dim = input.shape[static_cast<std::size_t>(rank - 1)];
  EXT_ENFORCE_INVALID(last_dim == 1 || last_dim == 2,
                      std::string(kDFTName) +
                          ": the last dimension of input must be 1 (real) or 2 (complex).");

  // Compute outer/inner factors. ``inner`` excludes the trailing real/imag dim.
  int64_t outer = 1;
  for (int64_t d = 0; d < a; ++d) {
    outer *= input.shape[static_cast<std::size_t>(d)];
  }
  int64_t inner = 1;
  for (int64_t d = a + 1; d < rank - 1; ++d) {
    inner *= input.shape[static_cast<std::size_t>(d)];
  }
  const int64_t in_axis = input.shape[static_cast<std::size_t>(a)];

  // Determine dft_length (n_dft).
  int64_t n_dft = in_axis;
  if (dft_length != nullptr) {
    n_dft = ReadDFTLength(*dft_length);
    EXT_ENFORCE_INVALID(n_dft > 0, std::string(kDFTName) + ": dft_length must be positive.");
  } else if (onesided && inverse) {
    // IRFFT default: 2 * (signal_dim_axis - 1).
    n_dft = 2 * (in_axis - 1);
    EXT_ENFORCE_INVALID(n_dft > 0,
                        std::string(kDFTName) + ": invalid default dft_length for IRFFT.");
  }

  // Determine the output axis dimension and trailing dimension.
  int64_t out_axis = n_dft;
  int64_t out_last = 2;
  if (onesided) {
    if (inverse) {
      // IRFFT: real-valued output of length n_dft.
      out_last = 1;
    } else {
      // RFFT: one-sided complex of length floor(n_dft / 2) + 1.
      out_axis = (n_dft / 2) + 1;
    }
  }

  // Build output shape: same rank as input. Axis dim becomes out_axis; trailing
  // dim becomes out_last.
  std::vector<int64_t> out_shape = input.shape;
  out_shape[static_cast<std::size_t>(a)] = out_axis;
  out_shape[static_cast<std::size_t>(rank - 1)] = out_last;

  int64_t out_total = 1;
  for (int64_t d : out_shape) {
    out_total *= d;
  }

  Tensor output(
      "", input.data_type, out_shape,
      std::vector<uint8_t>(static_cast<std::size_t>(out_total) * ElementSize(input.data_type)));

  switch (input.data_type) {
  case DataType::FLOAT:
    DftCompute<float>(input.AsFloat(), output.AsFloat(), outer, in_axis, out_axis, inner, last_dim,
                      out_last, n_dft, inverse, onesided);
    break;
  case DataType::DOUBLE:
    DftCompute<double>(input.AsDouble(), output.AsDouble(), outer, in_axis, out_axis, inner,
                       last_dim, out_last, n_dft, inverse, onesided);
    break;
  default:
    throw std::invalid_argument(std::string(kDFTName) +
                                " only supports FLOAT and DOUBLE input tensors.");
  }
  return output;
}

void DFT::operator()(const Tensor &input, const Tensor *dft_length, int64_t axis, bool onesided,
                     bool inverse, Tensor &output) const {
  Tensor produced = (*this)(input, dft_length, axis, onesided, inverse);
  EXT_ENFORCE_INVALID(output.data_type == produced.data_type,
                      std::string(kDFTName) + ": preallocated output dtype must match.");
  EXT_ENFORCE_INVALID(output.shape == produced.shape,
                      std::string(kDFTName) + ": preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(output.data.size() == produced.data.size(),
                      std::string(kDFTName) + ": preallocated output buffer size mismatch.");
  if (!produced.data.empty()) {
    std::memcpy(output.data.data(), produced.data.data(), produced.data.size());
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
