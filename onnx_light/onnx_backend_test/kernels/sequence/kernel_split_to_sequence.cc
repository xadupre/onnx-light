// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Reads the entries of ``split`` (an INT32 or INT64 tensor) into a
// vector of int64 values.
std::vector<int64_t> ReadSplit(const Tensor &split) {
  const int64_t n = split.shape.empty() ? int64_t{1} : [&]() {
    int64_t total = 1;
    for (int64_t d : split.shape)
      total *= d;
    return total;
  }();
  std::vector<int64_t> out;
  out.reserve(static_cast<std::size_t>(n));
  if (split.data_type == static_cast<int32_t>(DataType::INT32)) {
    const int32_t *p = split.AsInt32();
    EXT_ENFORCE_INVALID(p != nullptr || n == 0,
                        "kernel::SplitToSequence: 'split' INT32 data is null.");
    for (int64_t i = 0; i < n; ++i)
      out.push_back(static_cast<int64_t>(p[i]));
  } else {
    EXT_ENFORCE_INVALID(split.data_type == static_cast<int32_t>(DataType::INT64),
                        "kernel::SplitToSequence: 'split' must have data type INT32 or INT64.");
    const int64_t *p = split.AsInt64();
    EXT_ENFORCE_INVALID(p != nullptr || n == 0,
                        "kernel::SplitToSequence: 'split' INT64 data is null.");
    for (int64_t i = 0; i < n; ++i)
      out.push_back(p[i]);
  }
  return out;
}

// Resolves the per-output split sizes. Mirrors ONNX SplitToSequence:
//   * ``split`` omitted: ``axis_dim`` chunks of size 1.
//   * ``split`` is a scalar ``s``: equal chunks of size ``s``; the last
//     chunk takes the remainder when ``axis_dim`` is not divisible by ``s``.
//   * ``split`` is a 1-D tensor: its entries give the chunk sizes and
//     must sum to ``axis_dim``.
std::vector<int64_t> ResolveSplitSizes(int64_t axis_dim, const Tensor *split) {
  if (split == nullptr) {
    return std::vector<int64_t>(static_cast<std::size_t>(axis_dim), int64_t{1});
  }
  const std::vector<int64_t> values = ReadSplit(*split);
  if (split->shape.empty()) {
    // Scalar: split into equal chunks of ``values[0]``.
    EXT_ENFORCE_INVALID(values.size() == 1,
                        "kernel::SplitToSequence: scalar 'split' must contain exactly one value.");
    const int64_t chunk = values[0];
    EXT_ENFORCE_INVALID(chunk > 0,
                        "kernel::SplitToSequence: scalar 'split' must be strictly positive.");
    std::vector<int64_t> sizes;
    int64_t remaining = axis_dim;
    while (remaining > 0) {
      const int64_t take = remaining >= chunk ? chunk : remaining;
      sizes.push_back(take);
      remaining -= take;
    }
    if (sizes.empty()) {
      sizes.push_back(0);
    }
    return sizes;
  }
  // 1-D: use entries as-is.
  EXT_ENFORCE_INVALID(split->shape.size() == 1,
                      "kernel::SplitToSequence: 'split' must be a scalar or 1-D tensor.");
  int64_t total = 0;
  for (int64_t s : values) {
    EXT_ENFORCE_INVALID(s >= 0, "kernel::SplitToSequence: 'split' entries must be non-negative.");
    total += s;
  }
  EXT_ENFORCE_INVALID(total == axis_dim, "kernel::SplitToSequence: sum of 'split' (" +
                                             std::to_string(total) +
                                             ") does not match the input dim on 'axis' (" +
                                             std::to_string(axis_dim) + ").");
  return values;
}

} // namespace

Sequence SplitToSequence::operator()(const Tensor &input, const Tensor *split, int64_t axis,
                                     int64_t keepdims) const {
  const int64_t rank = static_cast<int64_t>(input.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, "kernel::SplitToSequence cannot split a scalar tensor.");

  const int64_t resolved_axis = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved_axis >= 0 && resolved_axis < rank,
                      "kernel::SplitToSequence axis is out of range.");

  const int64_t axis_dim = input.shape[static_cast<std::size_t>(resolved_axis)];
  const std::vector<int64_t> sizes = ResolveSplitSizes(axis_dim, split);

  // When ``split`` is provided, the schema mandates ``keepdims`` is ignored.
  const bool squeeze = (split == nullptr) && (keepdims == 0);

  const std::size_t elem_size = ElementSize(input.data_type);

  int64_t outer = 1;
  for (int64_t d = 0; d < resolved_axis; ++d) {
    outer *= input.shape[static_cast<std::size_t>(d)];
  }
  int64_t inner = 1;
  for (int64_t d = resolved_axis + 1; d < rank; ++d) {
    inner *= input.shape[static_cast<std::size_t>(d)];
  }
  const std::size_t inner_bytes = static_cast<std::size_t>(inner) * elem_size;
  const std::size_t in_row_bytes = static_cast<std::size_t>(axis_dim) * inner_bytes;

  std::vector<Tensor> outputs;
  outputs.reserve(sizes.size());
  std::size_t offset = 0; // byte offset within each "row" of the input.
  for (int64_t size : sizes) {
    std::vector<int64_t> out_shape;
    out_shape.reserve(static_cast<std::size_t>(rank));
    for (int64_t d = 0; d < rank; ++d) {
      if (d == resolved_axis) {
        if (!squeeze) {
          out_shape.push_back(size);
        }
        // When squeezing the axis must have size 1 by construction
        // (``split == nullptr`` implies all chunks have size 1).
      } else {
        out_shape.push_back(input.shape[static_cast<std::size_t>(d)]);
      }
    }
    int64_t total = 1;
    for (int64_t d : out_shape) {
      total *= d;
    }
    Tensor out("", input.data_type, out_shape,
               std::vector<uint8_t>(static_cast<std::size_t>(total) * elem_size));
    const std::size_t out_row_bytes = static_cast<std::size_t>(size) * inner_bytes;
    for (int64_t o = 0; o < outer; ++o) {
      std::memcpy(out.data.data() + static_cast<std::size_t>(o) * out_row_bytes,
                  input.data.data() + static_cast<std::size_t>(o) * in_row_bytes + offset,
                  out_row_bytes);
    }
    offset += out_row_bytes;
    outputs.push_back(std::move(out));
  }
  return Sequence("", input.data_type, std::move(outputs));
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
