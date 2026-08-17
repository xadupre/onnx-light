// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_extensions/kernels/kernels/nn/pool_attrs.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Row-major strides for ``shape``. Each stride is the number of elements one
// must skip to advance by one along that dimension.
Shape RowMajorStrides(const Shape &shape) {
  Shape strides;
  strides.assign(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Computes the size of the output along a single spatial axis according to
// the ONNX ``AveragePool`` formula with explicit padding. When
// ``ceil_mode`` is enabled, sliding windows that would start entirely in
// the right padded region are ignored (this matches ONNX Runtime and the
// ONNX reference implementation). ``dilation`` defaults to 1 (no
// dilation); for dilated kernels the effective kernel extent along an
// axis is ``dilation * (kernel - 1) + 1``.
int64_t OutputDim(int64_t in_dim, int64_t kernel, int64_t stride, int64_t pad_begin,
                  int64_t pad_end, bool ceil_mode, int64_t dilation = 1) {
  const int64_t eff_kernel = dilation * (kernel - 1) + 1;
  const double numerator =
      static_cast<double>(in_dim + pad_begin + pad_end - eff_kernel) / static_cast<double>(stride);
  const double v = ceil_mode ? std::ceil(numerator) : std::floor(numerator);
  int64_t out = static_cast<int64_t>(v) + 1;
  if (ceil_mode && out > 0) {
    // Drop the last window if its start position lies entirely in the right
    // padded region (i.e. start index >= in_dim in the padded coordinate
    // system).
    const int64_t last_start = (out - 1) * stride - pad_begin;
    if (last_start >= in_dim) {
      --out;
    }
  }
  return out;
}

// Resolves ``auto_pad`` into concrete output dimensions and explicit
// begin/end pads for one spatial axis. ``auto_pad`` must be one of
// ``SAME_UPPER``, ``SAME_LOWER`` or ``VALID``; ``NOTSET`` is handled by
// the caller using ``OutputDim`` and the explicit ``pads`` attribute.
void ResolveAutoPadAxis(AutoPad auto_pad, int64_t in_dim, int64_t kernel, int64_t stride,
                        int64_t dilation, int64_t &out_dim, int64_t &pad_begin, int64_t &pad_end) {
  const int64_t eff_kernel = dilation * (kernel - 1) + 1;
  if (auto_pad == AutoPad::kValid) {
    pad_begin = 0;
    pad_end = 0;
    const double numerator = static_cast<double>(in_dim - eff_kernel) / static_cast<double>(stride);
    out_dim = static_cast<int64_t>(std::floor(numerator)) + 1;
    return;
  }
  // SAME_UPPER / SAME_LOWER: output size = ceil(in_dim / stride).
  out_dim =
      static_cast<int64_t>(std::ceil(static_cast<double>(in_dim) / static_cast<double>(stride)));
  if (out_dim < 0) {
    out_dim = 0;
  }
  const int64_t pad_total = std::max<int64_t>(0, (out_dim - 1) * stride + eff_kernel - in_dim);
  if (auto_pad == AutoPad::kSameUpper) {
    pad_begin = pad_total / 2;
    pad_end = pad_total - pad_begin;
  } else { // SAME_LOWER
    pad_end = pad_total / 2;
    pad_begin = pad_total - pad_end;
  }
}

} // namespace

Tensor AveragePool::operator()(const Tensor &x, const Shape &kernel_shape, const Shape &strides,
                               const Shape &pads, bool ceil_mode, bool count_include_pad,
                               const Shape &dilations, AutoPad auto_pad, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::AveragePool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(!kernel_shape.empty(),
                      "kernel::AveragePool: kernel_shape must be non-empty.");
  EXT_ENFORCE_INVALID(
      x.shape.size() == kernel_shape.size() + 2,
      "kernel::AveragePool: x must have rank == kernel_shape.size() + 2 (N, C, D1, ..., Dk).");
  const size_t k = kernel_shape.size();
  Shape eff_strides;
  if (strides.empty()) {
    eff_strides.assign(k, 1);
  } else {
    eff_strides = strides;
  }
  Shape eff_dilations;
  if (dilations.empty()) {
    eff_dilations.assign(k, 1);
  } else {
    eff_dilations = dilations;
  }
  EXT_ENFORCE_INVALID(
      eff_strides.size() == k,
      "kernel::AveragePool: strides must be empty or have one entry per spatial axis.");
  EXT_ENFORCE_INVALID(
      eff_dilations.size() == k,
      "kernel::AveragePool: dilations must be empty or have one entry per spatial axis.");
  const bool use_auto_pad = auto_pad != AutoPad::kNotSet;
  EXT_ENFORCE_INVALID(!use_auto_pad || pads.empty(),
                      "kernel::AveragePool: pads must be empty when auto_pad is not NOTSET.");
  Shape eff_pads;
  if (use_auto_pad) {
    eff_pads.assign(2 * k, 0);
  } else {
    if (pads.empty()) {
      eff_pads.assign(2 * k, 0);
    } else {
      eff_pads = pads;
    }
  }
  EXT_ENFORCE_INVALID(
      eff_pads.size() == 2 * k,
      "kernel::AveragePool: pads must be empty or have two entries per spatial axis "
      "(begins followed by ends).");
  for (size_t i = 0; i < k; ++i) {
    EXT_ENFORCE_INVALID(kernel_shape[i] > 0,
                        "kernel::AveragePool: kernel_shape entries must be positive.");
    EXT_ENFORCE_INVALID(eff_strides[i] > 0,
                        "kernel::AveragePool: strides entries must be positive.");
    EXT_ENFORCE_INVALID(eff_dilations[i] > 0,
                        "kernel::AveragePool: dilations entries must be positive.");
    EXT_ENFORCE_INVALID(eff_pads[i] >= 0 && eff_pads[i + k] >= 0,
                        "kernel::AveragePool: pads entries must be non-negative.");
  }

  Shape out_shape;
  out_shape.assign(x.shape.size(), 0);
  out_shape[0] = x.shape[0];
  out_shape[1] = x.shape[1];
  for (size_t i = 0; i < k; ++i) {
    if (use_auto_pad) {
      int64_t pb = 0, pe = 0, od = 0;
      ResolveAutoPadAxis(auto_pad, x.shape[i + 2], kernel_shape[i], eff_strides[i],
                         eff_dilations[i], od, pb, pe);
      eff_pads[i] = pb;
      eff_pads[i + k] = pe;
      out_shape[i + 2] = od;
    } else {
      out_shape[i + 2] = OutputDim(x.shape[i + 2], kernel_shape[i], eff_strides[i], eff_pads[i],
                                   eff_pads[i + k], ceil_mode, eff_dilations[i]);
    }
    EXT_ENFORCE_INVALID(out_shape[i + 2] > 0,
                        "kernel::AveragePool: computed output spatial dimension is non-positive.");
  }

  int64_t n_out = 1;
  for (int64_t d : out_shape) {
    n_out *= d;
  }
  const size_t out_n_bytes = static_cast<size_t>(n_out) * sizeof(float);
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes, nullptr);
  // Forward to the in-place overload with auto_pad already resolved into
  // explicit pads (so the in-place overload need not duplicate the
  // resolution logic).
  (*this)(x, kernel_shape, eff_strides, eff_pads, ceil_mode, count_include_pad, out, eff_dilations,
          AutoPad::kNotSet);
  return out;
}

void AveragePool::operator()(const Tensor &x, const Shape &kernel_shape, const Shape &strides,
                             const Shape &pads, bool ceil_mode, bool count_include_pad,
                             Tensor &output, const Shape &dilations, AutoPad auto_pad) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::AveragePool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::AveragePool: output must be FLOAT.");
  EXT_ENFORCE_INVALID(!kernel_shape.empty() && x.shape.size() == kernel_shape.size() + 2,
                      "kernel::AveragePool: x rank must equal kernel_shape.size() + 2.");
  const size_t k = kernel_shape.size();
  EXT_ENFORCE_INVALID(strides.size() == k,
                      "kernel::AveragePool: strides must have one entry per spatial axis.");
  EXT_ENFORCE_INVALID(auto_pad == AutoPad::kNotSet || auto_pad == AutoPad::kSameUpper ||
                          auto_pad == AutoPad::kSameLower || auto_pad == AutoPad::kValid,
                      "kernel::AveragePool: auto_pad must be NOTSET, SAME_UPPER, SAME_LOWER "
                      "or VALID.");
  const bool use_auto_pad = auto_pad != AutoPad::kNotSet;
  EXT_ENFORCE_INVALID(!use_auto_pad || pads.empty(),
                      "kernel::AveragePool: pads must be empty when auto_pad is not NOTSET.");
  Shape eff_dilations;
  if (dilations.empty()) {
    eff_dilations.assign(k, 1);
  } else {
    eff_dilations = dilations;
  }
  EXT_ENFORCE_INVALID(
      eff_dilations.size() == k,
      "kernel::AveragePool: dilations must be empty or have one entry per spatial axis.");
  Shape eff_pads;
  if (use_auto_pad) {
    eff_pads.assign(2 * k, 0);
    for (size_t i = 0; i < k; ++i) {
      int64_t pb = 0, pe = 0, od = 0;
      ResolveAutoPadAxis(auto_pad, x.shape[i + 2], kernel_shape[i], strides[i], eff_dilations[i],
                         od, pb, pe);
      eff_pads[i] = pb;
      eff_pads[i + k] = pe;
    }
  } else {
    eff_pads = pads;
  }
  EXT_ENFORCE_INVALID(eff_pads.size() == 2 * k,
                      "kernel::AveragePool: pads must have two entries per spatial axis "
                      "(begins followed by ends).");
  EXT_ENFORCE_INVALID(output.shape.size() == x.shape.size(),
                      "kernel::AveragePool preallocated output rank must match x rank.");
  EXT_ENFORCE_INVALID(output.shape[0] == x.shape[0] && output.shape[1] == x.shape[1],
                      "kernel::AveragePool preallocated output N and C dimensions must match x.");
  for (size_t i = 0; i < k; ++i) {
    int64_t expected;
    if (use_auto_pad) {
      int64_t pb = 0, pe = 0;
      ResolveAutoPadAxis(auto_pad, x.shape[i + 2], kernel_shape[i], strides[i], eff_dilations[i],
                         expected, pb, pe);
    } else {
      expected = OutputDim(x.shape[i + 2], kernel_shape[i], strides[i], eff_pads[i],
                           eff_pads[i + k], ceil_mode, eff_dilations[i]);
    }
    EXT_ENFORCE_INVALID(
        output.shape[i + 2] == expected,
        "kernel::AveragePool preallocated output spatial dimension does not match the "
        "ONNX-computed value.");
  }
  int64_t n_out = 1;
  for (int64_t d : output.shape) {
    n_out *= d;
  }
  EXT_ENFORCE_INVALID(
      output.size_bytes() == static_cast<size_t>(n_out) * sizeof(float),
      "kernel::AveragePool preallocated output buffer has unexpected size in bytes.");

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(output.mutable_bytes());

  const Shape in_strides = RowMajorStrides(x.shape);
  const Shape out_strides = RowMajorStrides(output.shape);

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  // Iterate over (n, c) and then over the k-D spatial output grid using a
  // row-major counter over ``out_spatial_dims``.
  Shape out_spatial;
  out_spatial.assign(k, 0);
  for (size_t i = 0; i < k; ++i) {
    out_spatial[i] = output.shape[i + 2];
  }
  int64_t spatial_out_count = 1;
  for (int64_t d : out_spatial) {
    spatial_out_count *= d;
  }

  Shape out_idx;
  out_idx.assign(k, 0);
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
        //
        // ONNX semantics: a kernel position contributes to the divisor only
        // when it falls inside the padded input region
        // ``[-pad_begin, in_dim + pad_end)``.  Positions in
        // ``[0, in_dim)`` (``in_input``) contribute their real value;
        // positions in the padded region but outside the input
        // (``in_padded_region && !in_input``) contribute 0; positions
        // outside the padded region entirely (``ceil_mode`` overshoot or
        // otherwise) contribute nothing and are not counted in either
        // divisor.
        //
        // - ``count_include_pad=true``: divisor = number of positions in
        //   ``in_padded_region`` (i.e. ``in_window_count``).
        // - ``count_include_pad=false``: divisor = number of positions in
        //   ``in_input`` (i.e. ``valid_count``).
        double sum = 0.0;
        int64_t valid_count = 0;     // positions in [0, in_dim) (real values).
        int64_t in_window_count = 0; // positions in [-pad_begin, in_dim + pad_end).
        // Recursively (here: iteratively) iterate over the kernel volume.
        const int64_t kernel_volume = [&]() {
          int64_t v = 1;
          for (size_t i = 0; i < k; ++i) {
            v *= kernel_shape[i];
          }
          return v;
        }();
        Shape kidx;
        kidx.assign(k, 0);
        for (int64_t kflat = 0; kflat < kernel_volume; ++kflat) {
          int64_t krem = kflat;
          for (size_t i = k; i-- > 0;) {
            kidx[i] = krem % kernel_shape[i];
            krem /= kernel_shape[i];
          }
          int64_t in_offset = in_base;
          bool in_input = true;
          bool in_padded_region = true;
          for (size_t i = 0; i < k; ++i) {
            const int64_t p = out_idx[i] * strides[i] + kidx[i] * eff_dilations[i] - eff_pads[i];
            if (p < -eff_pads[i] || p >= x.shape[i + 2] + eff_pads[i + k]) {
              in_padded_region = false;
              in_input = false;
              break;
            }
            if (p < 0 || p >= x.shape[i + 2]) {
              in_input = false;
            } else {
              in_offset += p * in_strides[i + 2];
            }
          }
          if (in_padded_region) {
            ++in_window_count;
            if (in_input) {
              sum += static_cast<double>(px[in_offset]);
              ++valid_count;
            }
          }
        }
        int64_t denom = count_include_pad ? in_window_count : valid_count;
        int64_t out_offset = out_base;
        for (size_t i = 0; i < k; ++i) {
          out_offset += out_idx[i] * out_strides[i + 2];
        }
        py[out_offset] = denom == 0 ? 0.0f : static_cast<float>(sum / static_cast<double>(denom));
      }
    }
  }
}

void AveragePool::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const PoolCommonAttrs a = ParsePoolCommonAttrs(node);
  const bool count_include_pad = GetAttributeIntOrDefault(node, "count_include_pad", 0) != 0;
  onnx_kernels::kernel::AveragePool k(rt.kernel_ctx());
  SetOutput(node, 0,
            k(x, a.kernel_shape, a.strides, a.pads, a.ceil_mode, count_include_pad, a.dilations,
              a.auto_pad, &rt),
            rt.tensors());
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
