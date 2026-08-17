// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Reads a single INT32/INT64 scalar ``split`` value.
int64_t ReadScalarSplit(const Tensor &split) {
  if (split.data_type == static_cast<int32_t>(DataType::INT32)) {
    const int32_t *p = split.AsInt32();
    EXT_ENFORCE_INVALID(p != nullptr, "kernel::SplitToSequence: 'split' INT32 data is null.");
    return static_cast<int64_t>(p[0]);
  }
  EXT_ENFORCE_INVALID(split.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::SplitToSequence: 'split' must have data type INT32 or INT64.");
  const int64_t *p = split.AsInt64();
  EXT_ENFORCE_INVALID(p != nullptr, "kernel::SplitToSequence: 'split' INT64 data is null.");
  return p[0];
}

// Reads ``count`` entries of a 1-D INT32/INT64 ``split`` tensor into ``out``.
void ReadSplitInto(const Tensor &split, std::size_t count, int64_t *out) {
  if (split.data_type == static_cast<int32_t>(DataType::INT32)) {
    const int32_t *p = split.AsInt32();
    EXT_ENFORCE_INVALID(p != nullptr || count == 0,
                        "kernel::SplitToSequence: 'split' INT32 data is null.");
    for (std::size_t i = 0; i < count; ++i)
      out[i] = static_cast<int64_t>(p[i]);
  } else {
    EXT_ENFORCE_INVALID(split.data_type == static_cast<int32_t>(DataType::INT64),
                        "kernel::SplitToSequence: 'split' must have data type INT32 or INT64.");
    const int64_t *p = split.AsInt64();
    EXT_ENFORCE_INVALID(p != nullptr || count == 0,
                        "kernel::SplitToSequence: 'split' INT64 data is null.");
    for (std::size_t i = 0; i < count; ++i)
      out[i] = p[i];
  }
}

// Returns the number of chunks SplitToSequence produces. See ``FillSplitSizes``
// for the per-case semantics.
int64_t CountSplitSizes(int64_t axis_dim, const Tensor *split) {
  if (split == nullptr) {
    return axis_dim;
  }
  if (split->shape.empty()) {
    // Scalar: equal chunks of ``chunk``; an empty axis still yields a single
    // (zero-length) chunk.
    const int64_t chunk = ReadScalarSplit(*split);
    EXT_ENFORCE_INVALID(chunk > 0,
                        "kernel::SplitToSequence: scalar 'split' must be strictly positive.");
    return axis_dim > 0 ? (axis_dim + chunk - 1) / chunk : 1;
  }
  EXT_ENFORCE_INVALID(split->shape.size() == 1,
                      "kernel::SplitToSequence: 'split' must be a scalar or 1-D tensor.");
  return split->shape[0];
}

// Fills ``sizes`` (length == ``CountSplitSizes(axis_dim, split)``) with the
// per-output split sizes. Mirrors ONNX SplitToSequence:
//   * ``split`` omitted: ``axis_dim`` chunks of size 1.
//   * ``split`` is a scalar ``s``: equal chunks of size ``s``; the last
//     chunk takes the remainder when ``axis_dim`` is not divisible by ``s``.
//   * ``split`` is a 1-D tensor: its entries give the chunk sizes and
//     must sum to ``axis_dim``.
void FillSplitSizes(int64_t axis_dim, const Tensor *split, int64_t *sizes) {
  if (split == nullptr) {
    for (int64_t i = 0; i < axis_dim; ++i) {
      sizes[static_cast<std::size_t>(i)] = 1;
    }
    return;
  }
  if (split->shape.empty()) {
    const int64_t chunk = ReadScalarSplit(*split);
    // Empty axis yields a single zero-length chunk (``CountSplitSizes`` returned
    // 1), so ``sizes[0]`` is a valid slot here.
    if (axis_dim <= 0) {
      sizes[0] = 0;
      return;
    }
    std::size_t i = 0;
    int64_t remaining = axis_dim;
    while (remaining > 0) {
      const int64_t take = remaining >= chunk ? chunk : remaining;
      sizes[i++] = take;
      remaining -= take;
    }
    return;
  }
  // 1-D: use entries as-is.
  const std::size_t count = static_cast<std::size_t>(split->shape[0]);
  ReadSplitInto(*split, count, sizes);
  int64_t total = 0;
  for (std::size_t i = 0; i < count; ++i) {
    EXT_ENFORCE_INVALID(sizes[i] >= 0,
                        "kernel::SplitToSequence: 'split' entries must be non-negative.");
    total += sizes[i];
  }
  EXT_ENFORCE_INVALID(total == axis_dim, "kernel::SplitToSequence: sum of 'split' (",
                      std::to_string(total), ") does not match the input dim on 'axis' (",
                      std::to_string(axis_dim), ").");
}

} // namespace

Sequence SplitToSequence::operator()(const Tensor &input, const Tensor *split, int64_t axis,
                                     int64_t keepdims, RuntimeContext *rt) const {
  const int64_t rank = static_cast<int64_t>(input.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, "kernel::SplitToSequence cannot split a scalar tensor.");

  const int64_t resolved_axis = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved_axis >= 0 && resolved_axis < rank,
                      "kernel::SplitToSequence axis is out of range.");

  const int64_t axis_dim = input.shape[static_cast<std::size_t>(resolved_axis)];
  RawBufferAllocator *allocator = rt ? rt->execution_allocator() : nullptr;

  // ``sizes`` holds one entry per output chunk; its length scales with the
  // number of outputs, so it is drawn from the runtime allocator when one is
  // available, falling back to a ``std::vector`` otherwise.
  const int64_t num_sizes = CountSplitSizes(axis_dim, split);
  // The allocator rejects a zero-byte request; clamp to at least one slot.
  // ``num_sizes`` is 0 only for an omitted ``split`` with an empty axis, in
  // which case ``FillSplitSizes`` writes nothing and the loop below never runs.
  detail::TemporaryTypedBuffer<int64_t> sizes_buf(
      static_cast<std::size_t>(num_sizes > 0 ? num_sizes : 1), allocator,
      "kernel::SplitToSequence sizes");
  int64_t *sizes = sizes_buf.data();
  FillSplitSizes(axis_dim, split, sizes);

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

  Tensors outputs;
  outputs.reserve(static_cast<std::size_t>(num_sizes));
  std::size_t offset = 0; // byte offset within each "row" of the input.
  for (int64_t si = 0; si < num_sizes; ++si) {
    const int64_t size = sizes[static_cast<std::size_t>(si)];
    onnx_kernels::Shape out_shape;
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
    const size_t out_n_bytes = static_cast<std::size_t>(total) * elem_size;
    Tensor out = rt ? rt->MakeOutputTensor(0, input.data_type, out_shape, out_n_bytes)
                    : MakeOutputTensor(input.data_type, out_shape, out_n_bytes, nullptr);
    const std::size_t out_row_bytes = static_cast<std::size_t>(size) * inner_bytes;
    for (int64_t o = 0; o < outer; ++o) {
      std::memcpy(out.mutable_bytes() + static_cast<std::size_t>(o) * out_row_bytes,
                  input.bytes() + static_cast<std::size_t>(o) * in_row_bytes + offset,
                  out_row_bytes);
    }
    offset += out_row_bytes;
    outputs.push_back(std::move(out));
  }
  return Sequence("", input.data_type, std::move(outputs));
}

void SplitToSequence::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  EXT_ENFORCE_INVALID(!(node.input_size() > 2), "RunNode: op '", node.op_type(),
                      "' expects 1 or 2 inputs, got ", node.input_size(), ".");
  RequireOutputCount(node, 1);
  const Tensor &input = GetInput(node, 0, rt.tensors());
  const Tensor *split = GetOptionalInput(node, 1, rt.tensors());
  const int64_t axis = GetAttributeIntOrDefault(node, "axis", 0);
  const int64_t keepdims = GetAttributeIntOrDefault(node, "keepdims", 1);
  onnx_kernels::kernel::SplitToSequence k(rt.kernel_ctx());
  SetOutputSequence(node, 0, k(input, split, axis, keepdims), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
