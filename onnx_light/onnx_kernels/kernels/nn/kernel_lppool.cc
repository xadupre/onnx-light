// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"

#include <algorithm>
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
std::vector<int64_t> RowMajorStridesLpPool(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Computes the size of the output along a single spatial axis according to
// the ONNX ``LpPool`` formula with explicit padding. Mirrors
// :cpp:func:`AveragePool::OutputDim` — when ``ceil_mode`` is enabled,
// sliding windows that would start entirely in the right padded region are
// ignored (matching ONNX Runtime and the ONNX reference implementation).
int64_t OutputDimLpPool(int64_t in_dim, int64_t kernel, int64_t stride, int64_t pad_begin,
                        int64_t pad_end, bool ceil_mode, int64_t dilation) {
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

// Resolves ``auto_pad`` into concrete output dimensions and explicit
// begin/end pads for one spatial axis. ``auto_pad`` must be one of
// ``SAME_UPPER``, ``SAME_LOWER`` or ``VALID``.
void ResolveAutoPadAxisLpPool(const std::string &auto_pad, int64_t in_dim, int64_t kernel,
                              int64_t stride, int64_t dilation, int64_t &out_dim,
                              int64_t &pad_begin, int64_t &pad_end) {
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

} // namespace

Tensor LpPool::operator()(const Tensor &x, const std::vector<int64_t> &kernel_shape,
                          const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                          int64_t p, bool ceil_mode, const std::vector<int64_t> &dilations,
                          const std::string &auto_pad) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::LpPool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(!kernel_shape.empty(), "kernel::LpPool: kernel_shape must be non-empty.");
  EXT_ENFORCE_INVALID(
      x.shape.size() == kernel_shape.size() + 2,
      "kernel::LpPool: x must have rank == kernel_shape.size() + 2 (N, C, D1, ..., Dk).");
  EXT_ENFORCE_INVALID(p >= 1, "kernel::LpPool: p must be >= 1.");
  const size_t k = kernel_shape.size();
  std::vector<int64_t> eff_strides = strides.empty() ? std::vector<int64_t>(k, 1) : strides;
  std::vector<int64_t> eff_dilations = dilations.empty() ? std::vector<int64_t>(k, 1) : dilations;
  EXT_ENFORCE_INVALID(eff_strides.size() == k,
                      "kernel::LpPool: strides must be empty or have one entry per spatial axis.");
  EXT_ENFORCE_INVALID(
      eff_dilations.size() == k,
      "kernel::LpPool: dilations must be empty or have one entry per spatial axis.");
  EXT_ENFORCE_INVALID(auto_pad == "NOTSET" || auto_pad == "SAME_UPPER" ||
                          auto_pad == "SAME_LOWER" || auto_pad == "VALID",
                      "kernel::LpPool: auto_pad must be NOTSET, SAME_UPPER, SAME_LOWER or VALID.");
  const bool use_auto_pad = auto_pad != "NOTSET";
  EXT_ENFORCE_INVALID(!use_auto_pad || pads.empty(),
                      "kernel::LpPool: pads must be empty when auto_pad is not NOTSET.");
  std::vector<int64_t> eff_pads;
  if (use_auto_pad) {
    eff_pads.assign(2 * k, 0);
  } else {
    eff_pads = pads.empty() ? std::vector<int64_t>(2 * k, 0) : pads;
  }
  EXT_ENFORCE_INVALID(eff_pads.size() == 2 * k,
                      "kernel::LpPool: pads must be empty or have two entries per spatial axis "
                      "(begins followed by ends).");
  for (size_t i = 0; i < k; ++i) {
    EXT_ENFORCE_INVALID(kernel_shape[i] > 0,
                        "kernel::LpPool: kernel_shape entries must be positive.");
    EXT_ENFORCE_INVALID(eff_strides[i] > 0, "kernel::LpPool: strides entries must be positive.");
    EXT_ENFORCE_INVALID(eff_dilations[i] > 0,
                        "kernel::LpPool: dilations entries must be positive.");
    EXT_ENFORCE_INVALID(eff_pads[i] >= 0 && eff_pads[i + k] >= 0,
                        "kernel::LpPool: pads entries must be non-negative.");
  }

  std::vector<int64_t> out_shape(x.shape.size());
  out_shape[0] = x.shape[0];
  out_shape[1] = x.shape[1];
  for (size_t i = 0; i < k; ++i) {
    if (use_auto_pad) {
      int64_t pb = 0, pe = 0, od = 0;
      ResolveAutoPadAxisLpPool(auto_pad, x.shape[i + 2], kernel_shape[i], eff_strides[i],
                               eff_dilations[i], od, pb, pe);
      eff_pads[i] = pb;
      eff_pads[i + k] = pe;
      out_shape[i + 2] = od;
    } else {
      out_shape[i + 2] = OutputDimLpPool(x.shape[i + 2], kernel_shape[i], eff_strides[i],
                                         eff_pads[i], eff_pads[i + k], ceil_mode, eff_dilations[i]);
    }
    EXT_ENFORCE_INVALID(out_shape[i + 2] > 0,
                        "kernel::LpPool: computed output spatial dimension is non-positive.");
  }

  int64_t n_out = 1;
  for (int64_t d : out_shape) {
    n_out *= d;
  }
  Tensor out("", static_cast<int32_t>(DataType::FLOAT), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(n_out) * sizeof(float)));

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.data.data());

  const std::vector<int64_t> in_strides = RowMajorStridesLpPool(x.shape);
  const std::vector<int64_t> out_strides = RowMajorStridesLpPool(out.shape);

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  std::vector<int64_t> out_spatial(k);
  for (size_t i = 0; i < k; ++i) {
    out_spatial[i] = out.shape[i + 2];
  }
  int64_t spatial_out_count = 1;
  for (int64_t d : out_spatial) {
    spatial_out_count *= d;
  }

  const int64_t kernel_volume = [&]() {
    int64_t v = 1;
    for (size_t i = 0; i < k; ++i) {
      v *= kernel_shape[i];
    }
    return v;
  }();

  const double pd = static_cast<double>(p);
  const double inv_p = 1.0 / pd;

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
        double sum = 0.0;
        for (int64_t kflat = 0; kflat < kernel_volume; ++kflat) {
          int64_t krem = kflat;
          for (size_t i = k; i-- > 0;) {
            kidx[i] = krem % kernel_shape[i];
            krem /= kernel_shape[i];
          }
          int64_t in_offset = in_base;
          bool in_input = true;
          for (size_t i = 0; i < k; ++i) {
            const int64_t pos =
                out_idx[i] * eff_strides[i] + kidx[i] * eff_dilations[i] - eff_pads[i];
            if (pos < 0 || pos >= x.shape[i + 2]) {
              in_input = false;
              break;
            }
            in_offset += pos * in_strides[i + 2];
          }
          if (in_input) {
            const double v = std::fabs(static_cast<double>(px[in_offset]));
            sum += std::pow(v, pd);
          }
          // Positions outside the input contribute |0|^p = 0 (zero padding).
        }
        int64_t out_offset = out_base;
        for (size_t i = 0; i < k; ++i) {
          out_offset += out_idx[i] * out_strides[i + 2];
        }
        py[out_offset] = static_cast<float>(std::pow(sum, inv_p));
      }
    }
  }
  return out;
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
