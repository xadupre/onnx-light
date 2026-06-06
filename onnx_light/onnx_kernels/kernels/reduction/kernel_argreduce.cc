// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/reduction/include_reduction_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Resolves a possibly-negative axis (ONNX semantics: ``axis`` in
// ``[-rank, rank - 1]``) to a non-negative axis. Throws on out-of-range.
int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank, "kernel::ArgReduce: axis is out of range.");
  return resolved;
}

// Output shape of an argmax/argmin reduction along ``axis``: the reduced
// dimension is either dropped (``keepdims=false``) or replaced by 1.
std::vector<int64_t> ArgReduceOutputShape(const std::vector<int64_t> &in_shape, int64_t axis,
                                          bool keepdims) {
  std::vector<int64_t> out;
  out.reserve(in_shape.size());
  for (int64_t d = 0; d < static_cast<int64_t>(in_shape.size()); ++d) {
    if (d == axis) {
      if (keepdims) {
        out.push_back(1);
      }
    } else {
      out.push_back(in_shape[static_cast<size_t>(d)]);
    }
  }
  return out;
}

void ValidateFloat(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::FLOAT),
                      std::string("kernel::ArgReduce: ") + name + " must be a FLOAT tensor.");
}

void ValidateInt64(const Tensor &t, const char *name) {
  EXT_ENFORCE_INVALID(t.data_type == static_cast<int32_t>(DataType::INT64),
                      std::string("kernel::ArgReduce: ") + name + " must be an INT64 tensor.");
}

} // namespace

Tensor ArgReduce::operator()(const Tensor &data, int64_t axis, bool keepdims,
                             bool select_last_index) const {
  ValidateFloat(data, "data");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, "kernel::ArgReduce: data must have at least one dimension.");
  const int64_t resolved_axis = ResolveAxis(axis, rank);

  const std::vector<int64_t> out_shape = ArgReduceOutputShape(data.shape, resolved_axis, keepdims);
  int64_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= d;
  }
  Tensor out("", static_cast<int32_t>(DataType::INT64), out_shape,
             std::vector<uint8_t>(static_cast<size_t>(out_count) * sizeof(int64_t), 0u));
  (*this)(data, axis, keepdims, select_last_index, out);
  return out;
}

void ArgReduce::operator()(const Tensor &data, int64_t axis, bool keepdims, bool select_last_index,
                           Tensor &output) const {
  ValidateFloat(data, "data");
  ValidateInt64(output, "output");
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, "kernel::ArgReduce: data must have at least one dimension.");
  const int64_t resolved_axis = ResolveAxis(axis, rank);

  const std::vector<int64_t> expected_out_shape =
      ArgReduceOutputShape(data.shape, resolved_axis, keepdims);
  EXT_ENFORCE_INVALID(output.shape == expected_out_shape,
                      "kernel::ArgReduce preallocated output shape does not match expected.");
  const int64_t out_count = output.element_count();
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(out_count) * sizeof(int64_t),
                      "kernel::ArgReduce preallocated output buffer has unexpected size.");

  // Decompose the input layout into: outer (dims before axis), the reduced
  // ``axis_size`` dim, inner (dims after axis). The output is laid out as
  // ``outer * inner`` int64 values (same in either ``keepdims`` mode because
  // the byte buffer is identical; only the ``shape`` vector differs).
  int64_t outer = 1;
  for (int64_t d = 0; d < resolved_axis; ++d) {
    outer *= data.shape[static_cast<size_t>(d)];
  }
  const int64_t axis_size = data.shape[static_cast<size_t>(resolved_axis)];
  int64_t inner = 1;
  for (int64_t d = resolved_axis + 1; d < rank; ++d) {
    inner *= data.shape[static_cast<size_t>(d)];
  }
  EXT_ENFORCE_INVALID(axis_size > 0, "kernel::ArgReduce: cannot reduce along an axis of size 0.");

  const float *px = data.AsFloat();
  int64_t *py = output.AsInt64();
  const bool is_max = (mode_ == Mode::kMax);

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      const float *base = px + o * axis_size * inner + i;
      float best = base[0];
      int64_t best_idx = 0;
      for (int64_t k = 1; k < axis_size; ++k) {
        const float v = base[k * inner];
        bool replace;
        if (select_last_index) {
          // Keep the index of the last occurrence of the extremum.
          replace = is_max ? (v >= best) : (v <= best);
        } else {
          replace = is_max ? (v > best) : (v < best);
        }
        if (replace) {
          best = v;
          best_idx = k;
        }
      }
      py[o * inner + i] = best_idx;
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
