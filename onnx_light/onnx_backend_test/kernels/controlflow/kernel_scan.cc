// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Resolves a possibly-negative axis to a non-negative axis in [0, rank].
int64_t ResolveAxis(int64_t axis, std::size_t rank) {
  const int64_t r = static_cast<int64_t>(rank);
  if (axis < 0) {
    axis += r;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis <= r,
                      "kernel::Scan: axis is out of range for the output rank.");
  return axis;
}

// Returns the element-byte width for a populated tensor (data.size() /
// element_count()) or 0 when the tensor is empty.
std::size_t ElementBytes(const Tensor &t) {
  const int64_t n = t.element_count();
  if (n == 0) {
    return 0;
  }
  EXT_ENFORCE_INVALID(t.data.size() % static_cast<std::size_t>(n) == 0,
                      "kernel::Scan: tensor data is not a multiple of its element count.");
  return t.data.size() / static_cast<std::size_t>(n);
}

// Stacks the first ``trip_count`` per-iteration tensors along the chosen
// ``axis``, optionally reversing them when ``reverse`` is true (matching the
// ONNX Scan "prepend" direction).
Tensor StackScanOutput(const std::vector<Tensor> &per_iter, int64_t trip_count, int64_t axis_raw,
                       bool reverse) {
  EXT_ENFORCE_INVALID(static_cast<int64_t>(per_iter.size()) >= trip_count,
                      "kernel::Scan: scan-output row is shorter than the trip count.");

  DataType dtype = DataType::UNDEFINED;
  std::vector<int64_t> elt_shape;
  std::size_t elt_bytes = 0;
  if (!per_iter.empty()) {
    const Tensor &first = per_iter[0];
    dtype = static_cast<DataType>(first.data_type);
    elt_shape = first.shape;
    elt_bytes = ElementBytes(first);
  }
  for (int64_t t = 1; t < trip_count; ++t) {
    const Tensor &it = per_iter[static_cast<std::size_t>(t)];
    EXT_ENFORCE_INVALID(it.data_type == dtype,
                        "kernel::Scan: scan-output tensors must share the same data type "
                        "across iterations.");
    EXT_ENFORCE_INVALID(it.shape == elt_shape,
                        "kernel::Scan: scan-output tensors must share the same shape "
                        "across iterations.");
  }

  const std::size_t out_rank = elt_shape.size() + 1;
  const int64_t axis = ResolveAxis(axis_raw, elt_shape.size());

  std::vector<int64_t> out_shape;
  out_shape.reserve(out_rank);
  for (std::size_t d = 0; d <= elt_shape.size(); ++d) {
    if (static_cast<int64_t>(d) == axis) {
      out_shape.push_back(trip_count);
    }
    if (d < elt_shape.size()) {
      out_shape.push_back(elt_shape[d]);
    }
  }

  std::vector<uint8_t> out_data;
  if (trip_count > 0 && elt_bytes > 0) {
    // Element count of out_shape = trip_count * product(elt_shape).
    const int64_t elt_count =
        std::accumulate(elt_shape.begin(), elt_shape.end(), int64_t{1}, std::multiplies<int64_t>());
    const int64_t total_elts = trip_count * elt_count;
    out_data.resize(static_cast<std::size_t>(total_elts) * elt_bytes);

    // outer = product of out_shape[0..axis-1] in the *output*; for each
    // outer index we copy ``trip_count`` element-blocks of size
    // ``inner_bytes`` from consecutive per-iteration tensors. Because the
    // first ``axis`` dims of the output are exactly the first ``axis``
    // dims of the per-iteration tensor (the new axis is inserted *at*
    // position ``axis``), the per-iteration slice that contributes to
    // output index ``(outer, t, inner)`` lives at element offset
    // ``outer * inner + inner`` in the per-iteration tensor, where
    // ``inner`` is the product of dims at positions ``[axis..elt_rank)``.
    int64_t outer = 1;
    for (int64_t d = 0; d < axis; ++d) {
      outer *= elt_shape[static_cast<std::size_t>(d)];
    }
    int64_t inner = 1;
    for (std::size_t d = static_cast<std::size_t>(axis); d < elt_shape.size(); ++d) {
      inner *= elt_shape[d];
    }
    const std::size_t inner_bytes = static_cast<std::size_t>(inner) * elt_bytes;

    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t t = 0; t < trip_count; ++t) {
        const int64_t src_t = reverse ? (trip_count - 1 - t) : t;
        const Tensor &src = per_iter[static_cast<std::size_t>(src_t)];
        const std::size_t src_offset = static_cast<std::size_t>(o * inner) * elt_bytes;
        const std::size_t dst_offset =
            static_cast<std::size_t>(o * trip_count * inner + t * inner) * elt_bytes;
        std::memcpy(out_data.data() + dst_offset, src.data.data() + src_offset, inner_bytes);
      }
    }
  }
  return Tensor("", static_cast<int32_t>(dtype), out_shape, std::move(out_data));
}

} // namespace

std::vector<Tensor> Scan::operator()(int64_t trip_count, const std::vector<Tensor> &initial_state,
                                     const std::vector<Tensor> &final_state,
                                     const std::vector<std::vector<Tensor>> &scan_values_per_iter,
                                     const std::vector<int64_t> &scan_output_axes,
                                     const std::vector<int64_t> &scan_output_directions) const {
  EXT_ENFORCE_INVALID(trip_count >= 0, "kernel::Scan: trip_count must be non-negative.");
  EXT_ENFORCE_INVALID(initial_state.size() == final_state.size(),
                      "kernel::Scan: 'final_state' must have the same number of tensors "
                      "as 'initial_state'.");
  for (std::size_t i = 0; i < initial_state.size(); ++i) {
    EXT_ENFORCE_INVALID(initial_state[i].data_type == final_state[i].data_type,
                        "kernel::Scan: 'final_state' tensor data type must match the "
                        "corresponding 'initial_state' tensor.");
  }
  const std::size_t k = scan_values_per_iter.size();
  EXT_ENFORCE_INVALID(scan_output_axes.empty() || scan_output_axes.size() == k,
                      "kernel::Scan: 'scan_output_axes' must have K entries when provided.");
  EXT_ENFORCE_INVALID(scan_output_directions.empty() || scan_output_directions.size() == k,
                      "kernel::Scan: 'scan_output_directions' must have K entries when provided.");

  std::vector<Tensor> out;
  out.reserve(initial_state.size() + k);
  // N final state values: when the scan ran zero iterations, the final
  // value equals the initial value; otherwise it is whatever the caller
  // produced.
  for (std::size_t i = 0; i < initial_state.size(); ++i) {
    out.push_back(trip_count == 0 ? initial_state[i] : final_state[i]);
  }
  // K stacked scan outputs.
  for (std::size_t ki = 0; ki < k; ++ki) {
    const int64_t axis = scan_output_axes.empty() ? 0 : scan_output_axes[ki];
    const bool reverse = !scan_output_directions.empty() && scan_output_directions[ki] != 0;
    out.push_back(StackScanOutput(scan_values_per_iter[ki], trip_count, axis, reverse));
  }
  return out;
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
