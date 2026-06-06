// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Row-major strides for ``shape``.
std::vector<int64_t> RowMajorStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Mirrors ``kernel_averagepool.cc::OutputDim``: computes the output size
// along a single spatial axis with explicit padding and optional ceil_mode.
int64_t OutputDim(int64_t in_dim, int64_t kernel, int64_t stride, int64_t pad_begin,
                  int64_t pad_end, bool ceil_mode, int64_t dilation = 1) {
  const int64_t eff_kernel = dilation * (kernel - 1) + 1;
  const double numerator =
      static_cast<double>(in_dim + pad_begin + pad_end - eff_kernel) / static_cast<double>(stride);
  const double v = ceil_mode ? std::ceil(numerator) : std::floor(numerator);
  int64_t out = static_cast<int64_t>(v) + 1;
  if (ceil_mode && out > 0) {
    const int64_t last_start = (out - 1) * stride - pad_begin;
    if (last_start >= in_dim) {
      --out;
    }
  }
  return out;
}

// Mirrors ``kernel_averagepool.cc::ResolveAutoPadAxis``.
void ResolveAutoPadAxis(const std::string &auto_pad, int64_t in_dim, int64_t kernel, int64_t stride,
                        int64_t dilation, int64_t &out_dim, int64_t &pad_begin, int64_t &pad_end) {
  const int64_t eff_kernel = dilation * (kernel - 1) + 1;
  if (auto_pad == "VALID") {
    pad_begin = 0;
    pad_end = 0;
    const double numerator = static_cast<double>(in_dim - eff_kernel) / static_cast<double>(stride);
    out_dim = static_cast<int64_t>(std::floor(numerator)) + 1;
    return;
  }
  out_dim =
      static_cast<int64_t>(std::ceil(static_cast<double>(in_dim) / static_cast<double>(stride)));
  if (out_dim < 0) {
    out_dim = 0;
  }
  const int64_t pad_total = std::max<int64_t>(0, (out_dim - 1) * stride + eff_kernel - in_dim);
  if (auto_pad == "SAME_UPPER") {
    pad_begin = pad_total / 2;
    pad_end = pad_total - pad_begin;
  } else { // SAME_LOWER
    pad_end = pad_total / 2;
    pad_begin = pad_total - pad_end;
  }
}

// Validates inputs and resolves output spatial shape and effective pads,
// producing both Y and (optionally) Indices. ``produce_indices`` controls
// whether the second output buffer is allocated and populated.
std::pair<Tensor, Tensor> RunMaxPool(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                                     const std::vector<int64_t> &strides_in,
                                     const std::vector<int64_t> &pads_in, bool ceil_mode,
                                     const std::vector<int64_t> &dilations_in,
                                     int64_t storage_order, const std::string &auto_pad,
                                     bool produce_indices) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::MaxPool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(!kernel_shape.empty(), "kernel::MaxPool: kernel_shape must be non-empty.");
  EXT_ENFORCE_INVALID(
      x.shape.size() == kernel_shape.size() + 2,
      "kernel::MaxPool: x must have rank == kernel_shape.size() + 2 (N, C, D1, ..., Dk).");
  EXT_ENFORCE_INVALID(storage_order == 0,
                      "kernel::MaxPool: only storage_order=0 (row major) is supported.");

  const size_t k = kernel_shape.size();
  std::vector<int64_t> strides = strides_in.empty() ? std::vector<int64_t>(k, 1) : strides_in;
  std::vector<int64_t> dilations = dilations_in.empty() ? std::vector<int64_t>(k, 1) : dilations_in;
  EXT_ENFORCE_INVALID(strides.size() == k,
                      "kernel::MaxPool: strides must have one entry per spatial axis.");
  EXT_ENFORCE_INVALID(dilations.size() == k,
                      "kernel::MaxPool: dilations must have one entry per spatial axis.");
  EXT_ENFORCE_INVALID(auto_pad == "NOTSET" || auto_pad == "SAME_UPPER" ||
                          auto_pad == "SAME_LOWER" || auto_pad == "VALID",
                      "kernel::MaxPool: auto_pad must be NOTSET, SAME_UPPER, SAME_LOWER or VALID.");
  const bool use_auto_pad = auto_pad != "NOTSET";
  EXT_ENFORCE_INVALID(!use_auto_pad || pads_in.empty(),
                      "kernel::MaxPool: pads must be empty when auto_pad is not NOTSET.");

  std::vector<int64_t> pads;
  if (use_auto_pad) {
    pads.assign(2 * k, 0);
  } else {
    pads = pads_in.empty() ? std::vector<int64_t>(2 * k, 0) : pads_in;
  }
  EXT_ENFORCE_INVALID(pads.size() == 2 * k,
                      "kernel::MaxPool: pads must have 2 * k entries (begins then ends).");
  for (size_t i = 0; i < k; ++i) {
    EXT_ENFORCE_INVALID(kernel_shape[i] > 0,
                        "kernel::MaxPool: kernel_shape entries must be positive.");
    EXT_ENFORCE_INVALID(strides[i] > 0, "kernel::MaxPool: strides entries must be positive.");
    EXT_ENFORCE_INVALID(dilations[i] > 0, "kernel::MaxPool: dilations entries must be positive.");
    EXT_ENFORCE_INVALID(pads[i] >= 0 && pads[i + k] >= 0,
                        "kernel::MaxPool: pads entries must be non-negative.");
  }

  std::vector<int64_t> out_shape(x.shape.size());
  out_shape[0] = x.shape[0];
  out_shape[1] = x.shape[1];
  for (size_t i = 0; i < k; ++i) {
    if (use_auto_pad) {
      int64_t pb = 0, pe = 0, od = 0;
      ResolveAutoPadAxis(auto_pad, x.shape[i + 2], kernel_shape[i], strides[i], dilations[i], od,
                         pb, pe);
      pads[i] = pb;
      pads[i + k] = pe;
      out_shape[i + 2] = od;
    } else {
      out_shape[i + 2] = OutputDim(x.shape[i + 2], kernel_shape[i], strides[i], pads[i],
                                   pads[i + k], ceil_mode, dilations[i]);
    }
    EXT_ENFORCE_INVALID(out_shape[i + 2] > 0,
                        "kernel::MaxPool: computed output spatial dimension is non-positive.");
  }

  int64_t n_out = 1;
  for (int64_t d : out_shape) {
    n_out *= d;
  }
  Tensor y("", static_cast<int32_t>(DataType::FLOAT), out_shape,
           std::vector<uint8_t>(static_cast<size_t>(n_out) * sizeof(float)));
  Tensor indices;
  if (produce_indices) {
    indices = Tensor("", static_cast<int32_t>(DataType::INT64), out_shape,
                     std::vector<uint8_t>(static_cast<size_t>(n_out) * sizeof(int64_t)));
  }

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(y.data.data());
  int64_t *pi = produce_indices ? reinterpret_cast<int64_t *>(indices.data.data()) : nullptr;

  const std::vector<int64_t> in_strides = RowMajorStrides(x.shape);
  const std::vector<int64_t> out_strides = RowMajorStrides(out_shape);

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  std::vector<int64_t> out_spatial(k);
  for (size_t i = 0; i < k; ++i) {
    out_spatial[i] = out_shape[i + 2];
  }
  int64_t spatial_out_count = 1;
  for (int64_t d : out_spatial) {
    spatial_out_count *= d;
  }
  int64_t kernel_volume = 1;
  for (size_t i = 0; i < k; ++i) {
    kernel_volume *= kernel_shape[i];
  }

  std::vector<int64_t> out_idx(k);
  std::vector<int64_t> kidx(k);
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t in_base = n * in_strides[0] + c * in_strides[1];
      const int64_t out_base = n * out_strides[0] + c * out_strides[1];
      for (int64_t flat = 0; flat < spatial_out_count; ++flat) {
        int64_t rem = flat;
        for (size_t i = k; i-- > 0;) {
          out_idx[i] = rem % out_spatial[i];
          rem /= out_spatial[i];
        }
        float best = -std::numeric_limits<float>::infinity();
        int64_t best_in_offset = -1;
        bool any = false;
        for (int64_t kflat = 0; kflat < kernel_volume; ++kflat) {
          int64_t krem = kflat;
          for (size_t i = k; i-- > 0;) {
            kidx[i] = krem % kernel_shape[i];
            krem /= kernel_shape[i];
          }
          int64_t in_offset = in_base;
          bool in_input = true;
          for (size_t i = 0; i < k; ++i) {
            const int64_t p = out_idx[i] * strides[i] + kidx[i] * dilations[i] - pads[i];
            if (p < 0 || p >= x.shape[i + 2]) {
              in_input = false;
              break;
            }
            in_offset += p * in_strides[i + 2];
          }
          if (in_input) {
            const float v = px[in_offset];
            if (!any || v > best) {
              best = v;
              best_in_offset = in_offset;
              any = true;
            }
          }
        }
        int64_t out_offset = out_base;
        for (size_t i = 0; i < k; ++i) {
          out_offset += out_idx[i] * out_strides[i + 2];
        }
        py[out_offset] = any ? best : -std::numeric_limits<float>::infinity();
        if (produce_indices) {
          pi[out_offset] = best_in_offset;
        }
      }
    }
  }
  return {std::move(y), std::move(indices)};
}

} // namespace

Tensor MaxPool::operator()(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                           const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                           bool ceil_mode, const std::vector<int64_t> &dilations,
                           int64_t storage_order, const std::string &auto_pad) const {
  auto result = RunMaxPool(x, kernel_shape, strides, pads, ceil_mode, dilations, storage_order,
                           auto_pad, /*produce_indices=*/false);
  return std::move(result.first);
}

std::pair<Tensor, Tensor>
MaxPool::WithIndices(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                     const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                     bool ceil_mode, const std::vector<int64_t> &dilations, int64_t storage_order,
                     const std::string &auto_pad) const {
  return RunMaxPool(x, kernel_shape, strides, pads, ceil_mode, dilations, storage_order, auto_pad,
                    /*produce_indices=*/true);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
