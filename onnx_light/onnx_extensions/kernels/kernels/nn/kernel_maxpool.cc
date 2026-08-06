// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/nn/pool_attrs.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Row-major strides for ``shape``.
Shape RowMajorStrides(const Shape &shape) {
  Shape strides;
  strides.assign(shape.size(), 1);
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

// Per-type lowest representable value used as the initial ``best`` accumulator
// when scanning a pooling window. For floating point types we use ``-inf`` to
// match the upstream ONNX reference; for integer types we use the minimum of
// the type so that any in-window value wins on the first comparison.
template <typename T> constexpr T MaxPoolInitial() {
  if constexpr (std::numeric_limits<T>::has_infinity) {
    return -std::numeric_limits<T>::infinity();
  } else {
    return std::numeric_limits<T>::lowest();
  }
}

// Core pooling loop templated on the element type. Reads ``px`` and writes
// ``py``/``pi`` in row-major order using the strides/shapes already resolved
// by the caller. ``produce_indices`` controls whether ``pi`` is populated.
// ``index_spatial_strides`` are the per-axis strides used to flatten the
// selected spatial coordinate into the ``Indices`` value: row-major strides
// for ``storage_order == 0`` and column-major strides for
// ``storage_order == 1``.
template <typename T>
void MaxPoolLoop(const T *px, T *py, int64_t *pi, bool produce_indices, int64_t N, int64_t C,
                 const Shape &x_shape, const Shape &in_strides, const Shape &out_strides,
                 const Shape &out_spatial, const Shape &kernel_shape, const Shape &strides,
                 const Shape &dilations, const Shape &pads, const Shape &index_spatial_strides) {
  const size_t k = kernel_shape.size();
  int64_t spatial_out_count = 1;
  for (int64_t d : out_spatial) {
    spatial_out_count *= d;
  }
  int64_t kernel_volume = 1;
  for (size_t i = 0; i < k; ++i) {
    kernel_volume *= kernel_shape[i];
  }
  Shape out_idx;
  out_idx.assign(k, 0);
  Shape kidx;
  kidx.assign(k, 0);
  Shape best_p;
  best_p.assign(k, 0);
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
        T best = MaxPoolInitial<T>();
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
            if (p < 0 || p >= x_shape[i + 2]) {
              in_input = false;
              break;
            }
            kidx[i] = p;
            in_offset += p * in_strides[i + 2];
          }
          if (in_input) {
            const T v = px[in_offset];
            if (!any || v > best) {
              best = v;
              best_p = kidx;
              any = true;
            }
          }
        }
        int64_t out_offset = out_base;
        for (size_t i = 0; i < k; ++i) {
          out_offset += out_idx[i] * out_strides[i + 2];
        }
        py[out_offset] = any ? best : MaxPoolInitial<T>();
        if (produce_indices) {
          if (any) {
            int64_t index = in_base;
            for (size_t i = 0; i < k; ++i) {
              index += best_p[i] * index_spatial_strides[i];
            }
            pi[out_offset] = index;
          } else {
            pi[out_offset] = -1;
          }
        }
      }
    }
  }
}

// Validates inputs and resolves output spatial shape and effective pads,
// producing both Y and (optionally) Indices. ``produce_indices`` controls
// whether the second output buffer is allocated and populated.
std::pair<Tensor, Tensor> RunMaxPool(const Tensor &x, const Shape &kernel_shape,
                                     const Shape &strides_in, const Shape &pads_in, bool ceil_mode,
                                     const Shape &dilations_in, int64_t storage_order,
                                     AutoPad auto_pad, bool produce_indices,
                                     RawBufferAllocator *allocator = nullptr) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT) ||
                          x.data_type == static_cast<int32_t>(DataType::DOUBLE) ||
                          x.data_type == static_cast<int32_t>(DataType::INT8) ||
                          x.data_type == static_cast<int32_t>(DataType::UINT8),
                      "kernel::MaxPool: x must be FLOAT, DOUBLE, INT8 or UINT8.");
  EXT_ENFORCE_INVALID(!kernel_shape.empty(), "kernel::MaxPool: kernel_shape must be non-empty.");
  EXT_ENFORCE_INVALID(
      x.shape.size() == kernel_shape.size() + 2,
      "kernel::MaxPool: x must have rank == kernel_shape.size() + 2 (N, C, D1, ..., Dk).");
  EXT_ENFORCE_INVALID(storage_order == 0 || storage_order == 1,
                      "kernel::MaxPool: storage_order must be 0 (row major) or 1 (column major).");

  const size_t k = kernel_shape.size();
  Shape strides;
  if (strides_in.empty()) {
    strides.assign(k, 1);
  } else {
    strides = strides_in;
  }
  Shape dilations;
  if (dilations_in.empty()) {
    dilations.assign(k, 1);
  } else {
    dilations = dilations_in;
  }
  EXT_ENFORCE_INVALID(strides.size() == k,
                      "kernel::MaxPool: strides must have one entry per spatial axis.");
  EXT_ENFORCE_INVALID(dilations.size() == k,
                      "kernel::MaxPool: dilations must have one entry per spatial axis.");
  EXT_ENFORCE_INVALID(auto_pad == AutoPad::kNotSet || auto_pad == AutoPad::kSameUpper ||
                          auto_pad == AutoPad::kSameLower || auto_pad == AutoPad::kValid,
                      "kernel::MaxPool: auto_pad must be NOTSET, SAME_UPPER, SAME_LOWER or VALID.");
  const bool use_auto_pad = auto_pad != AutoPad::kNotSet;
  EXT_ENFORCE_INVALID(!use_auto_pad || pads_in.empty(),
                      "kernel::MaxPool: pads must be empty when auto_pad is not NOTSET.");

  Shape pads;
  if (use_auto_pad) {
    pads.assign(2 * k, 0);
  } else {
    if (pads_in.empty()) {
      pads.assign(2 * k, 0);
    } else {
      pads = pads_in;
    }
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

  Shape out_shape;
  out_shape.assign(x.shape.size(), 0);
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
  const size_t elem_size = ElementSize(x.data_type);
  const size_t y_n_bytes = static_cast<size_t>(n_out) * elem_size;
  Tensor y = MakeOutputTensor(x.data_type, out_shape, y_n_bytes, allocator);
  Tensor indices;
  if (produce_indices) {
    const size_t indices_n_bytes = static_cast<size_t>(n_out) * sizeof(int64_t);
    indices = MakeOutputTensor(static_cast<int32_t>(DataType::INT64), out_shape, indices_n_bytes,
                               allocator);
  }

  const Shape in_strides = RowMajorStrides(x.shape);
  const Shape out_strides = RowMajorStrides(out_shape);

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  Shape out_spatial;
  out_spatial.assign(k, 0);
  for (size_t i = 0; i < k; ++i) {
    out_spatial[i] = out_shape[i + 2];
  }

  // Per-axis spatial strides used to flatten the selected input coordinate
  // into the ``Indices`` value. ``storage_order == 0`` uses row-major strides
  // (matching ``in_strides`` over the spatial axes); ``storage_order == 1``
  // uses column-major strides over the spatial axes.
  Shape index_spatial_strides;
  index_spatial_strides.assign(k, 0);
  if (storage_order == 0) {
    for (size_t i = 0; i < k; ++i) {
      index_spatial_strides[i] = in_strides[i + 2];
    }
  } else {
    int64_t stride = 1;
    for (size_t i = 0; i < k; ++i) {
      index_spatial_strides[i] = stride;
      stride *= x.shape[i + 2];
    }
  }

  int64_t *pi = produce_indices ? reinterpret_cast<int64_t *>(indices.mutable_bytes()) : nullptr;

  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    MaxPoolLoop<float>(x.AsFloat(), reinterpret_cast<float *>(y.mutable_bytes()), pi,
                       produce_indices, N, C, x.shape, in_strides, out_strides, out_spatial,
                       kernel_shape, strides, dilations, pads, index_spatial_strides);
    break;
  case DataType::DOUBLE:
    MaxPoolLoop<double>(x.AsDouble(), reinterpret_cast<double *>(y.mutable_bytes()), pi,
                        produce_indices, N, C, x.shape, in_strides, out_strides, out_spatial,
                        kernel_shape, strides, dilations, pads, index_spatial_strides);
    break;
  case DataType::INT8:
    MaxPoolLoop<int8_t>(x.AsInt8(), reinterpret_cast<int8_t *>(y.mutable_bytes()), pi,
                        produce_indices, N, C, x.shape, in_strides, out_strides, out_spatial,
                        kernel_shape, strides, dilations, pads, index_spatial_strides);
    break;
  case DataType::UINT8:
    MaxPoolLoop<uint8_t>(x.AsUint8(), reinterpret_cast<uint8_t *>(y.mutable_bytes()), pi,
                         produce_indices, N, C, x.shape, in_strides, out_strides, out_spatial,
                         kernel_shape, strides, dilations, pads, index_spatial_strides);
    break;
  default:
    EXT_ENFORCE_INVALID(false, "kernel::MaxPool: unsupported element type.");
  }
  return {std::move(y), std::move(indices)};
}

} // namespace

Tensor MaxPool::operator()(const Tensor &x, const Shape &kernel_shape, const Shape &strides,
                           const Shape &pads, bool ceil_mode, const Shape &dilations,
                           int64_t storage_order, AutoPad auto_pad, RuntimeContext *rt) const {
  auto result = RunMaxPool(x, kernel_shape, strides, pads, ceil_mode, dilations, storage_order,
                           auto_pad, /*produce_indices=*/false, rt ? rt->allocator() : nullptr);
  return std::move(result.first);
}

std::pair<Tensor, Tensor> MaxPool::WithIndices(const Tensor &x, const Shape &kernel_shape,
                                               const Shape &strides, const Shape &pads,
                                               bool ceil_mode, const Shape &dilations,
                                               int64_t storage_order, AutoPad auto_pad) const {
  return RunMaxPool(x, kernel_shape, strides, pads, ceil_mode, dilations, storage_order, auto_pad,
                    /*produce_indices=*/true);
}

void MaxPool::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  EXT_ENFORCE_INVALID(!(node.output_size() < 1 || node.output_size() > 2),
                      "RunNode: op 'MaxPool' expects 1 or 2 output(s), got ", node.output_size(),
                      ".");
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const PoolCommonAttrs a = ParsePoolCommonAttrs(node);
  const int64_t storage_order = GetAttributeIntOrDefault(node, "storage_order", 0);
  onnx_kernels::kernel::MaxPool k(rt.kernel_ctx());
  const bool need_indices = node.output_size() == 2 && !node.output(1).empty();
  if (need_indices) {
    auto result = k.WithIndices(x, a.kernel_shape, a.strides, a.pads, a.ceil_mode, a.dilations,
                                storage_order, a.auto_pad);
    SetOutput(node, 0, std::move(result.first), rt.tensors());
    const std::string indices_name = node.output(1);
    result.second.name = indices_name;
    rt.tensors()[indices_name] = std::move(result.second);
  } else {
    SetOutput(node, 0,
              k(x, a.kernel_shape, a.strides, a.pads, a.ceil_mode, a.dilations, storage_order,
                a.auto_pad),
              rt.tensors());
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
