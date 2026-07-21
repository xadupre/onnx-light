// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/onnx_kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kTopKName = "kernel::TopK";

// Resolves a possibly-negative axis to a non-negative one in ``[0, rank)``.
int64_t ResolveTopKAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank, kTopKName, ": axis is out of range.");
  return resolved;
}

// Splits ``shape`` around ``axis`` into ``outer`` (product of dims before
// ``axis``), ``axis_dim`` (shape[axis]) and ``inner`` (product of dims after
// ``axis``).
void SplitTopKShape(const Shape &shape, int64_t axis, int64_t &outer, int64_t &axis_dim,
                    int64_t &inner) {
  outer = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= shape[static_cast<std::size_t>(i)];
  }
  axis_dim = shape[static_cast<std::size_t>(axis)];
  inner = 1;
  for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < shape.size(); ++i) {
    inner *= shape[i];
  }
}

template <typename T>
void TopKCompute(const Tensor &x, int64_t k, int64_t axis, bool largest, bool sorted, T *values_out,
                 int64_t *indices_out) {
  int64_t outer = 0;
  int64_t axis_dim = 0;
  int64_t inner = 0;
  SplitTopKShape(x.shape, axis, outer, axis_dim, inner);

  const T *px = x.As<T>();

  // Indices into ``[0, axis_dim)``; reused across slices.
  std::vector<int64_t> idx(static_cast<std::size_t>(axis_dim));

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t in = 0; in < inner; ++in) {
      std::iota(idx.begin(), idx.end(), int64_t{0});
      const int64_t base = (o * axis_dim) * inner + in;

      // Stable comparator over ``axis`` index; ties broken by smaller index
      // (matches the ONNX reference and the schema's "lower index appears
      // first" guarantee).
      auto less = [&](int64_t a, int64_t b) {
        const T va = px[base + a * inner];
        const T vb = px[base + b * inner];
        if (largest) {
          if (va != vb) {
            return va > vb;
          }
        } else {
          if (va != vb) {
            return va < vb;
          }
        }
        return a < b;
      };

      // Partition the top-k indices to the front of ``idx``. ``nth_element``
      // gives O(N) average partitioning.
      std::nth_element(idx.begin(), idx.begin() + k, idx.end(), less);
      // Always sort the prefix so that output ordering is deterministic.
      // (The ONNX schema does not mandate any ordering when ``sorted`` is 0,
      // but a deterministic result keeps tests reproducible.)
      std::sort(idx.begin(), idx.begin() + k, less);
      (void)sorted;

      const int64_t out_base = (o * k) * inner + in;
      for (int64_t j = 0; j < k; ++j) {
        const int64_t src = idx[static_cast<std::size_t>(j)];
        values_out[out_base + j * inner] = px[base + src * inner];
        indices_out[out_base + j * inner] = src;
      }
    }
  }
}

constexpr const char *kSupportedTopKTypesMsg = " only supports common numeric tensor types.";

template <typename T>
void DispatchTopK(const Tensor &x, int64_t k, int64_t axis, bool largest, bool sorted,
                  Tensor &values, Tensor &indices) {
  TopKCompute<T>(x, k, axis, largest, sorted, values.As<T>(), indices.As<int64_t>());
}

void RunTopK(const Tensor &x, int64_t k, int64_t axis, bool largest, bool sorted, Tensor &values,
             Tensor &indices) {
  switch (x.data_type) {
  case DataType::FLOAT:
    DispatchTopK<float>(x, k, axis, largest, sorted, values, indices);
    return;
  case DataType::DOUBLE:
    DispatchTopK<double>(x, k, axis, largest, sorted, values, indices);
    return;
  case DataType::INT8:
    DispatchTopK<int8_t>(x, k, axis, largest, sorted, values, indices);
    return;
  case DataType::INT16:
    DispatchTopK<int16_t>(x, k, axis, largest, sorted, values, indices);
    return;
  case DataType::INT32:
    DispatchTopK<int32_t>(x, k, axis, largest, sorted, values, indices);
    return;
  case DataType::INT64:
    DispatchTopK<int64_t>(x, k, axis, largest, sorted, values, indices);
    return;
  case DataType::UINT8:
    DispatchTopK<uint8_t>(x, k, axis, largest, sorted, values, indices);
    return;
  case DataType::UINT16:
    DispatchTopK<uint16_t>(x, k, axis, largest, sorted, values, indices);
    return;
  case DataType::UINT32:
    DispatchTopK<uint32_t>(x, k, axis, largest, sorted, values, indices);
    return;
  case DataType::UINT64:
    DispatchTopK<uint64_t>(x, k, axis, largest, sorted, values, indices);
    return;
  default:
    EXT_THROW_INVALID(kTopKName, ": unsupported data type ", x.data_type, kSupportedTopKTypesMsg);
  }
}

Shape MakeOutputShape(const Shape &shape, int64_t axis, int64_t k) {
  Shape out = shape;
  out[static_cast<std::size_t>(axis)] = k;
  return out;
}

} // namespace

std::pair<Tensor, Tensor> TopK::operator()(const Tensor &x, int64_t k, int64_t axis, bool largest,
                                           bool sorted, RuntimeContext *rt) const {
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, kTopKName, " requires a non-scalar input.");
  EXT_ENFORCE_INVALID(k > 0, kTopKName, " requires k > 0.");
  const int64_t resolved_axis = ResolveTopKAxis(axis, rank);
  EXT_ENFORCE_INVALID(k <= x.shape[static_cast<std::size_t>(resolved_axis)], kTopKName,
                      ": k is larger than the axis dimension.");

  const Shape out_shape = MakeOutputShape(x.shape, resolved_axis, k);
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  const int64_t out_count = [&] {
    int64_t n = 1;
    for (int64_t d : out_shape) {
      n *= d;
    }
    return n;
  }();
  Tensor values =
      MakeOutputTensor(x.data_type, out_shape,
                       static_cast<std::size_t>(out_count) * ElementSize(x.data_type), allocator);
  Tensor indices =
      MakeOutputTensor(static_cast<int32_t>(DataType::INT64), out_shape,
                       static_cast<std::size_t>(out_count) * sizeof(int64_t), allocator);
  RunTopK(x, k, resolved_axis, largest, sorted, values, indices);
  return std::pair<Tensor, Tensor>(std::move(values), std::move(indices));
}

void TopK::operator()(const Tensor &x, int64_t k, int64_t axis, bool largest, bool sorted,
                      Tensor &values, Tensor &indices) const {
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, kTopKName, " requires a non-scalar input.");
  EXT_ENFORCE_INVALID(k > 0, kTopKName, " requires k > 0.");
  const int64_t resolved_axis = ResolveTopKAxis(axis, rank);
  EXT_ENFORCE_INVALID(k <= x.shape[static_cast<std::size_t>(resolved_axis)], kTopKName,
                      ": k is larger than the axis dimension.");

  const Shape out_shape = MakeOutputShape(x.shape, resolved_axis, k);
  EXT_ENFORCE_INVALID(values.data_type == x.data_type, kTopKName,
                      " preallocated Values output must share the input dtype.");
  EXT_ENFORCE_INVALID(values.shape == out_shape, kTopKName,
                      " preallocated Values output shape does not match expected.");
  EXT_ENFORCE_INVALID(indices.data_type == static_cast<int32_t>(DataType::INT64), kTopKName,
                      " preallocated Indices output must have INT64 dtype.");
  EXT_ENFORCE_INVALID(indices.shape == out_shape, kTopKName,
                      " preallocated Indices output shape does not match expected.");
  RunTopK(x, k, resolved_axis, largest, sorted, values, indices);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
