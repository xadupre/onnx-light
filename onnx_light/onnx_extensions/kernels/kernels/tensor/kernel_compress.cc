// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Normalise ``axis`` into the range [0, rank) and throw if out of range.
int64_t NormaliseCompressAxis(int64_t axis, int64_t rank) {
  if (axis < 0) {
    axis += rank;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis < rank, "kernel::Compress: axis out of range.");
  return axis;
}

// Validate the 'condition' input and return a direct view over its bytes. The
// BOOL tensor stores one byte per element (0 == false, non-zero == true), so no
// intermediate copy (and therefore no working-memory allocation) is required.
const uint8_t *ValidateCondition(const Tensor &condition) {
  EXT_ENFORCE_INVALID(condition.data_type == static_cast<int32_t>(DataType::BOOL),
                      "kernel::Compress: 'condition' must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(condition.shape.size() == 1, "kernel::Compress: 'condition' must be rank-1.");
  return condition.bytes();
}

} // namespace

Tensor Compress::operator()(const Tensor &input, const Tensor &condition,
                            std::optional<int64_t> axis, RuntimeContext *rt) const {
  const uint8_t *cond = ValidateCondition(condition);
  const std::size_t cond_len = static_cast<std::size_t>(condition.element_count());
  const std::size_t elem_size = ElementSize(input.data_type);
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;

  if (!axis.has_value()) {
    // Flatten mode: select individual elements from the flattened input. The
    // selected elements are counted first, then copied in a single pass, so no
    // scratch index buffer is allocated at all.
    const int64_t total = input.element_count();
    int64_t out_count = 0;
    for (int64_t i = 0; i < total && static_cast<std::size_t>(i) < cond_len; ++i) {
      if (cond[static_cast<std::size_t>(i)]) {
        ++out_count;
      }
    }
    const size_t output_n_bytes = static_cast<std::size_t>(out_count) * elem_size;
    Tensor output = MakeOutputTensor(input.data_type, {out_count}, output_n_bytes, allocator);
    int64_t k = 0;
    for (int64_t i = 0; i < total && static_cast<std::size_t>(i) < cond_len; ++i) {
      if (cond[static_cast<std::size_t>(i)]) {
        std::memcpy(output.mutable_bytes() + static_cast<std::size_t>(k) * elem_size,
                    input.bytes() + static_cast<std::size_t>(i) * elem_size, elem_size);
        ++k;
      }
    }
    return output;
  }

  // Axis mode: select slices along the given axis.
  const int64_t rank = static_cast<int64_t>(input.shape.size());
  const int64_t a = NormaliseCompressAxis(*axis, rank);

  // Compute the number of selected slices (condition[i] == true for i < axis_dim).
  const int64_t axis_dim = input.shape[static_cast<std::size_t>(a)];
  int64_t selected_count = 0;
  for (int64_t i = 0; i < axis_dim && static_cast<std::size_t>(i) < cond_len; ++i) {
    if (cond[static_cast<std::size_t>(i)]) {
      ++selected_count;
    }
  }

  // Build output shape: same as input but with axis_dim replaced by selected_count.
  onnx_kernels::Shape out_shape = input.shape;
  out_shape[static_cast<std::size_t>(a)] = selected_count;

  // Number of elements in the output.
  int64_t out_total = 1;
  for (int64_t d : out_shape) {
    out_total *= d;
  }

  const size_t output_n_bytes = static_cast<std::size_t>(out_total) * elem_size;
  Tensor output = MakeOutputTensor(input.data_type, out_shape, output_n_bytes, allocator);

  if (out_total == 0) {
    return output;
  }

  // Compute the number of outer elements (dims before axis) and inner elements (dims after axis).
  int64_t outer = 1;
  for (int64_t d = 0; d < a; ++d) {
    outer *= input.shape[static_cast<std::size_t>(d)];
  }
  int64_t inner = 1;
  for (int64_t d = a + 1; d < rank; ++d) {
    inner *= input.shape[static_cast<std::size_t>(d)];
  }

  // Iterate over the selected slices and copy them in a single pass, tracking
  // the destination slice index directly instead of collecting the selected
  // axis indices into a scratch buffer.
  for (int64_t o = 0; o < outer; ++o) {
    int64_t s = 0;
    for (int64_t i = 0; i < axis_dim; ++i) {
      if (static_cast<std::size_t>(i) < cond_len && cond[static_cast<std::size_t>(i)]) {
        // Source flat offset for this (outer, axis_idx) block.
        const int64_t src_base = (o * axis_dim + i) * inner;
        const int64_t dst_base = (o * selected_count + s) * inner;
        std::memcpy(output.mutable_bytes() + static_cast<std::size_t>(dst_base) * elem_size,
                    input.bytes() + static_cast<std::size_t>(src_base) * elem_size,
                    static_cast<std::size_t>(inner) * elem_size);
        ++s;
      }
    }
  }
  return output;
}

void Compress::operator()(const Tensor &input, const Tensor &condition, std::optional<int64_t> axis,
                          Tensor &output) const {
  Tensor produced = (*this)(input, condition, axis);
  EXT_ENFORCE_INVALID(output.data_type == produced.data_type,
                      "kernel::Compress: preallocated output dtype must match.");
  EXT_ENFORCE_INVALID(output.shape == produced.shape,
                      "kernel::Compress: preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(output.size_bytes() == produced.size_bytes(),
                      "kernel::Compress: preallocated output buffer size mismatch.");
  if (!produced.data.empty()) {
    std::memcpy(output.mutable_bytes(), produced.bytes(), produced.size_bytes());
  }
}

void Compress::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &input = GetInput(node, 0, rt.tensors());
  const Tensor &condition = GetInput(node, 1, rt.tensors());
  const AttributeProto *axis_attr = FindAttribute(node, "axis");
  std::optional<int64_t> axis;
  if (axis_attr != nullptr) {
    axis = axis_attr->i();
  }
  onnx_kernels::kernel::Compress k(rt.kernel_ctx());
  SetOutput(node, 0, k(input, condition, axis, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
