// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Row-major strides for ``shape``. Each stride is the number of elements one
// must skip to advance by one along that dimension.
std::vector<int64_t> RowMajorStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Computes the size of the output along a single spatial axis according to
// the ONNX ``AveragePool`` formula.
int64_t OutputDim(int64_t in_dim, int64_t kernel, int64_t stride, int64_t pad_begin,
                  int64_t pad_end, bool ceil_mode) {
  const double numerator =
      static_cast<double>(in_dim + pad_begin + pad_end - kernel) / static_cast<double>(stride);
  const double v = ceil_mode ? std::ceil(numerator) : std::floor(numerator);
  return static_cast<int64_t>(v) + 1;
}

} // namespace

Tensor AveragePool::operator()(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                               const std::vector<int64_t> &strides,
                               const std::vector<int64_t> &pads, bool ceil_mode,
                               bool count_include_pad) const {
  if (x.data_type != static_cast<int32_t>(TensorProto::DataType::FLOAT)) {
    throw std::invalid_argument("kernel::AveragePool: x must be FLOAT.");
  }
  if (kernel_shape.empty()) {
    throw std::invalid_argument("kernel::AveragePool: kernel_shape must be non-empty.");
  }
  if (x.shape.size() != kernel_shape.size() + 2) {
    throw std::invalid_argument(
        "kernel::AveragePool: x must have rank == kernel_shape.size() + 2 (N, C, D1, ..., Dk).");
  }
  const size_t k = kernel_shape.size();
  std::vector<int64_t> eff_strides = strides.empty() ? std::vector<int64_t>(k, 1) : strides;
  std::vector<int64_t> eff_pads = pads.empty() ? std::vector<int64_t>(2 * k, 0) : pads;
  if (eff_strides.size() != k) {
    throw std::invalid_argument(
        "kernel::AveragePool: strides must be empty or have one entry per spatial axis.");
  }
  if (eff_pads.size() != 2 * k) {
    throw std::invalid_argument(
        "kernel::AveragePool: pads must be empty or have two entries per spatial axis "
        "(begins followed by ends).");
  }
  for (size_t i = 0; i < k; ++i) {
    if (kernel_shape[i] <= 0) {
      throw std::invalid_argument("kernel::AveragePool: kernel_shape entries must be positive.");
    }
    if (eff_strides[i] <= 0) {
      throw std::invalid_argument("kernel::AveragePool: strides entries must be positive.");
    }
    if (eff_pads[i] < 0 || eff_pads[i + k] < 0) {
      throw std::invalid_argument("kernel::AveragePool: pads entries must be non-negative.");
    }
  }

  std::vector<int64_t> out_shape(x.shape.size());
  out_shape[0] = x.shape[0];
  out_shape[1] = x.shape[1];
  for (size_t i = 0; i < k; ++i) {
    out_shape[i + 2] = OutputDim(x.shape[i + 2], kernel_shape[i], eff_strides[i], eff_pads[i],
                                 eff_pads[i + k], ceil_mode);
    if (out_shape[i + 2] <= 0) {
      throw std::invalid_argument(
          "kernel::AveragePool: computed output spatial dimension is non-positive.");
    }
  }

  int64_t n_out = 1;
  for (int64_t d : out_shape) {
    n_out *= d;
  }
  Tensor out("", static_cast<int32_t>(TensorProto::DataType::FLOAT), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(n_out) * sizeof(float)));
  (*this)(x, kernel_shape, eff_strides, eff_pads, ceil_mode, count_include_pad, out);
  return out;
}

void AveragePool::operator()(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                             const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                             bool ceil_mode, bool count_include_pad, Tensor &output) const {
  if (x.data_type != static_cast<int32_t>(TensorProto::DataType::FLOAT)) {
    throw std::invalid_argument("kernel::AveragePool: x must be FLOAT.");
  }
  if (output.data_type != static_cast<int32_t>(TensorProto::DataType::FLOAT)) {
    throw std::invalid_argument("kernel::AveragePool: output must be FLOAT.");
  }
  if (kernel_shape.empty() || x.shape.size() != kernel_shape.size() + 2) {
    throw std::invalid_argument("kernel::AveragePool: x rank must equal kernel_shape.size() + 2.");
  }
  const size_t k = kernel_shape.size();
  if (strides.size() != k) {
    throw std::invalid_argument(
        "kernel::AveragePool: strides must have one entry per spatial axis.");
  }
  if (pads.size() != 2 * k) {
    throw std::invalid_argument("kernel::AveragePool: pads must have two entries per spatial axis "
                                "(begins followed by ends).");
  }
  if (output.shape.size() != x.shape.size()) {
    throw std::invalid_argument("kernel::AveragePool preallocated output rank must match x rank.");
  }
  if (output.shape[0] != x.shape[0] || output.shape[1] != x.shape[1]) {
    throw std::invalid_argument(
        "kernel::AveragePool preallocated output N and C dimensions must match x.");
  }
  for (size_t i = 0; i < k; ++i) {
    const int64_t expected =
        OutputDim(x.shape[i + 2], kernel_shape[i], strides[i], pads[i], pads[i + k], ceil_mode);
    if (output.shape[i + 2] != expected) {
      throw std::invalid_argument(
          "kernel::AveragePool preallocated output spatial dimension does not match the "
          "ONNX-computed value.");
    }
  }
  int64_t n_out = 1;
  for (int64_t d : output.shape) {
    n_out *= d;
  }
  if (output.data.size() != static_cast<size_t>(n_out) * sizeof(float)) {
    throw std::invalid_argument(
        "kernel::AveragePool preallocated output buffer has unexpected size in bytes.");
  }

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(output.data.data());

  const std::vector<int64_t> in_strides = RowMajorStrides(x.shape);
  const std::vector<int64_t> out_strides = RowMajorStrides(output.shape);

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  // Iterate over (n, c) and then over the k-D spatial output grid using a
  // row-major counter over ``out_spatial_dims``.
  std::vector<int64_t> out_spatial(k);
  for (size_t i = 0; i < k; ++i) {
    out_spatial[i] = output.shape[i + 2];
  }
  int64_t spatial_out_count = 1;
  for (int64_t d : out_spatial) {
    spatial_out_count *= d;
  }

  std::vector<int64_t> out_idx(k);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t in_base = n * in_strides[0] + c * in_strides[1];
      const int64_t out_base = n * out_strides[0] + c * out_strides[1];
      for (int64_t flat = 0; flat < spatial_out_count; ++flat) {
        // Decode the flat output index into ``out_idx`` (row-major).
        int64_t rem = flat;
        for (size_t i = k; i-- > 0;) {
          out_idx[i] = rem % out_spatial[i];
          rem /= out_spatial[i];
        }
        // Accumulate the average over the kernel window.
        double sum = 0.0;
        int64_t valid_count = 0;
        // Recursively (here: iteratively) iterate over the kernel volume.
        const int64_t kernel_volume = [&]() {
          int64_t v = 1;
          for (size_t i = 0; i < k; ++i) {
            v *= kernel_shape[i];
          }
          return v;
        }();
        std::vector<int64_t> kidx(k);
        for (int64_t kflat = 0; kflat < kernel_volume; ++kflat) {
          int64_t krem = kflat;
          for (size_t i = k; i-- > 0;) {
            kidx[i] = krem % kernel_shape[i];
            krem /= kernel_shape[i];
          }
          int64_t in_offset = in_base;
          bool inside = true;
          for (size_t i = 0; i < k; ++i) {
            const int64_t p = out_idx[i] * strides[i] + kidx[i] - pads[i];
            if (p < 0 || p >= x.shape[i + 2]) {
              inside = false;
              break;
            }
            in_offset += p * in_strides[i + 2];
          }
          if (inside) {
            sum += static_cast<double>(px[in_offset]);
            ++valid_count;
          }
        }
        int64_t denom;
        if (count_include_pad) {
          denom = 1;
          for (size_t i = 0; i < k; ++i) {
            denom *= kernel_shape[i];
          }
        } else {
          denom = valid_count;
        }
        int64_t out_offset = out_base;
        for (size_t i = 0; i < k; ++i) {
          out_offset += out_idx[i] * out_strides[i + 2];
        }
        py[out_offset] = denom == 0 ? 0.0f : static_cast<float>(sum / static_cast<double>(denom));
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
