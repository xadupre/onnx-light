// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Resolves the per-output split sizes. Exactly one of ``split`` (non-empty)
// or ``num_outputs`` (> 0) must be set. Mirrors the ONNX Split semantics:
//   - When ``split`` is provided, its entries are used as-is and must sum
//     to ``axis_dim``.
//   - When ``num_outputs`` is provided, ``axis_dim`` is divided into equal
//     chunks of ``ceil(axis_dim / num_outputs)``; the last chunk absorbs the
//     remainder (and may be smaller).
std::vector<int64_t> ResolveSplitSizes(int64_t axis_dim, const std::vector<int64_t> &split,
                                       int64_t num_outputs) {
  if (!split.empty()) {
    EXT_ENFORCE_INVALID(num_outputs <= 0,
                        "kernel::Split: 'split' and 'num_outputs' are mutually exclusive.");
    int64_t total = 0;
    for (int64_t s : split) {
      EXT_ENFORCE_INVALID(s >= 0, "kernel::Split: 'split' entries must be non-negative.");
      total += s;
    }
    EXT_ENFORCE_INVALID(total == axis_dim, "kernel::Split: sum of 'split' (" +
                                               std::to_string(total) +
                                               ") does not match the input dim on 'axis' (" +
                                               std::to_string(axis_dim) + ").");
    return split;
  }
  EXT_ENFORCE_INVALID(num_outputs > 0,
                      "kernel::Split: either 'split' or 'num_outputs' must be specified.");
  // Per ONNX Split-18 spec: divide evenly; the last chunk takes the remainder
  // and may be smaller. ``chunk = ceil(axis_dim / num_outputs)``.
  const int64_t chunk = (axis_dim + num_outputs - 1) / num_outputs;
  std::vector<int64_t> sizes(static_cast<size_t>(num_outputs), chunk);
  int64_t remaining = axis_dim;
  for (size_t i = 0; i + 1 < sizes.size(); ++i) {
    remaining -= chunk;
  }
  // The last chunk takes whatever is left (may be 0 or smaller than chunk).
  sizes.back() = remaining < 0 ? 0 : remaining;
  return sizes;
}

} // namespace

std::vector<Tensor> Split::operator()(const Tensor &input, int64_t axis,
                                      const std::vector<int64_t> &split,
                                      int64_t num_outputs) const {
  const int64_t rank = static_cast<int64_t>(input.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, "kernel::Split cannot split a scalar tensor.");

  const int64_t resolved_axis = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved_axis >= 0 && resolved_axis < rank,
                      "kernel::Split axis is out of range.");

  const int64_t axis_dim = input.shape[static_cast<size_t>(resolved_axis)];
  const std::vector<int64_t> sizes = ResolveSplitSizes(axis_dim, split, num_outputs);

  const size_t elem_size = ElementSize(input.data_type);

  // ``outer`` is the product of dimensions before ``axis``; ``inner`` is the
  // product of dimensions after ``axis``. Each output is laid out as
  // ``outer`` contiguous rows of ``sizes[i] * inner`` elements taken from the
  // matching slice in ``input``.
  int64_t outer = 1;
  for (int64_t d = 0; d < resolved_axis; ++d) {
    outer *= input.shape[static_cast<size_t>(d)];
  }
  int64_t inner = 1;
  for (int64_t d = resolved_axis + 1; d < rank; ++d) {
    inner *= input.shape[static_cast<size_t>(d)];
  }
  const size_t inner_bytes = static_cast<size_t>(inner) * elem_size;
  const size_t in_row_bytes = static_cast<size_t>(axis_dim) * inner_bytes;

  std::vector<Tensor> outputs;
  outputs.reserve(sizes.size());
  size_t offset = 0; // byte offset within each "row" of the input.
  for (int64_t size : sizes) {
    std::vector<int64_t> out_shape = input.shape;
    out_shape[static_cast<size_t>(resolved_axis)] = size;
    int64_t total = 1;
    for (int64_t d : out_shape) {
      total *= d;
    }
    Tensor out("", input.data_type, out_shape,
               std::vector<uint8_t>(static_cast<size_t>(total) * elem_size));
    const size_t out_row_bytes = static_cast<size_t>(size) * inner_bytes;
    for (int64_t o = 0; o < outer; ++o) {
      std::memcpy(out.data.data() + static_cast<size_t>(o) * out_row_bytes,
                  input.bytes() + static_cast<size_t>(o) * in_row_bytes + offset, out_row_bytes);
    }
    offset += out_row_bytes;
    outputs.push_back(std::move(out));
  }
  return outputs;
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
