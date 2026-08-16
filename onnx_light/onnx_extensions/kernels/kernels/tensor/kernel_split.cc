// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Resolves the per-output split sizes. Exactly one of ``split`` (non-empty)
// or ``num_outputs`` (> 0) must be set. Mirrors the ONNX Split semantics:
//   - When ``split`` is provided, its entries are used as-is and must sum
//     to ``axis_dim``.
//   - When ``num_outputs`` is provided, ``axis_dim`` is divided into equal
//     chunks of ``ceil(axis_dim / num_outputs)``; the last chunk absorbs the
//     remainder (and may be smaller).
std::vector<int64_t> ResolveSplitSizes(int64_t axis_dim, std::span<const int64_t> split,
                                       int64_t num_outputs) {
  if (!split.empty()) {
    EXT_ENFORCE_INVALID(num_outputs <= 0,
                        "kernel::Split: 'split' and 'num_outputs' are mutually exclusive.");
    int64_t total = 0;
    for (int64_t s : split) {
      EXT_ENFORCE_INVALID(s >= 0, "kernel::Split: 'split' entries must be non-negative.");
      total += s;
    }
    EXT_ENFORCE_INVALID(total == axis_dim, "kernel::Split: sum of 'split' (", std::to_string(total),
                        ") does not match the input dim on 'axis' (", std::to_string(axis_dim),
                        ").");
    return std::vector<int64_t>(split.begin(), split.end());
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

Tensors Split::operator()(const Tensor &input, int64_t axis, std::span<const int64_t> split,
                          int64_t num_outputs, RuntimeContext *rt) const {
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

  Tensors outputs;
  outputs.reserve(sizes.size());
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  size_t offset = 0; // byte offset within each "row" of the input.
  for (int64_t size : sizes) {
    onnx_kernels::Shape out_shape = input.shape;
    out_shape[static_cast<size_t>(resolved_axis)] = size;
    int64_t total = 1;
    for (int64_t d : out_shape) {
      total *= d;
    }
    const size_t out_n_bytes = static_cast<size_t>(total) * elem_size;
    Tensor out = MakeOutputTensor(input.data_type, out_shape, out_n_bytes, allocator);
    const size_t out_row_bytes = static_cast<size_t>(size) * inner_bytes;
    for (int64_t o = 0; o < outer; ++o) {
      std::memcpy(out.mutable_bytes() + static_cast<size_t>(o) * out_row_bytes,
                  input.bytes() + static_cast<size_t>(o) * in_row_bytes + offset, out_row_bytes);
    }
    offset += out_row_bytes;
    outputs.push_back(std::move(out));
  }
  return outputs;
}

void Split::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  EXT_ENFORCE_INVALID(!(node.input_size() > 2), "RunNode: op '", node.op_type(),
                      "' expects 1 or 2 inputs, got ", node.input_size(), ".");
  EXT_ENFORCE_INVALID(!(node.output_size() < 1), "RunNode: op '", node.op_type(),
                      "' expects at least 1 output, got 0.");
  const Tensor &input = GetInput(node, 0, rt.tensors());
  const int64_t axis = GetAttributeIntOrDefault(node, "axis", 0);

  // Resolve ``split``: from the optional 2nd input (opset >= 13), from the
  // legacy ``split`` attribute (opset <= 12), or unspecified.
  // When the split sizes come from a tensor input, use a zero-copy span
  // view; otherwise fall back to an attribute-sourced vector.
  const Tensor *split_input = GetOptionalInput(node, 1, rt.tensors());
  std::vector<int64_t> split_attr;
  std::span<const int64_t> split;
  if (split_input != nullptr) {
    split = TensorSpan<int64_t>(*split_input);
  } else {
    split_attr = GetAttributeIntsOrDefault(node, "split", {});
    split = split_attr;
  }

  // ``num_outputs`` (opset >= 18) defaults to the number of outputs of the
  // node when neither ``split`` nor the attribute is provided.
  int64_t num_outputs = GetAttributeIntOrDefault(node, "num_outputs", 0);
  if (split.empty() && num_outputs <= 0) {
    num_outputs = static_cast<int64_t>(node.output_size());
  }

  onnx_kernels::kernel::Split k(rt.kernel_ctx());
  Tensors outputs = k(input, axis, split, num_outputs);
  EXT_ENFORCE_INVALID(!(static_cast<int>(outputs.size()) != node.output_size()),
                      "RunNode: op 'Split' produced ", outputs.size(),
                      " outputs but node declares ", node.output_size(), ".");
  for (int i = 0; i < node.output_size(); ++i) {
    SetOutput(node, i, std::move(outputs[static_cast<size_t>(i)]), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
