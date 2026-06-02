// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Normalise ``axis`` into the range [0, rank) and throw if out of range.
int64_t NormaliseCompressAxis(int64_t axis, int64_t rank) {
  if (axis < 0) {
    axis += rank;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis < rank, "kernel::Compress: axis out of range.");
  return axis;
}

// Read the boolean values from a BOOL tensor.  Returns true/false per element.
std::vector<bool> ReadCondition(const Tensor &condition) {
  EXT_ENFORCE_INVALID(condition.data_type == static_cast<int32_t>(DataType::BOOL),
                      "kernel::Compress: 'condition' must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(condition.shape.size() == 1, "kernel::Compress: 'condition' must be rank-1.");
  const int64_t n = condition.element_count();
  std::vector<bool> result(static_cast<std::size_t>(n));
  const uint8_t *ptr = condition.data.data();
  for (int64_t i = 0; i < n; ++i) {
    result[static_cast<std::size_t>(i)] = (ptr[static_cast<std::size_t>(i)] != 0);
  }
  return result;
}

} // namespace

Tensor Compress::operator()(const Tensor &input, const Tensor &condition,
                            std::optional<int64_t> axis) const {
  const std::vector<bool> cond = ReadCondition(condition);
  const std::size_t cond_len = cond.size();
  const std::size_t elem_size = ElementSize(input.data_type);

  if (!axis.has_value()) {
    // Flatten mode: select individual elements from the flattened input.
    const int64_t total = input.element_count();
    std::vector<int64_t> selected_indices;
    for (int64_t i = 0; i < total && static_cast<std::size_t>(i) < cond_len; ++i) {
      if (cond[static_cast<std::size_t>(i)]) {
        selected_indices.push_back(i);
      }
    }
    const int64_t out_count = static_cast<int64_t>(selected_indices.size());
    Tensor output("", input.data_type, {out_count},
                  std::vector<uint8_t>(static_cast<std::size_t>(out_count) * elem_size));
    for (int64_t k = 0; k < out_count; ++k) {
      const std::size_t src_off =
          static_cast<std::size_t>(selected_indices[static_cast<std::size_t>(k)]) * elem_size;
      const std::size_t dst_off = static_cast<std::size_t>(k) * elem_size;
      std::memcpy(output.data.data() + dst_off, input.data.data() + src_off, elem_size);
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
  std::vector<int64_t> out_shape = input.shape;
  out_shape[static_cast<std::size_t>(a)] = selected_count;

  // Compute strides (in elements) for the input tensor.
  std::vector<int64_t> strides(static_cast<std::size_t>(rank), 1);
  for (int64_t d = rank - 2; d >= 0; --d) {
    strides[static_cast<std::size_t>(d)] =
        strides[static_cast<std::size_t>(d + 1)] * input.shape[static_cast<std::size_t>(d + 1)];
  }

  // Number of elements in the output.
  int64_t out_total = 1;
  for (int64_t d : out_shape) {
    out_total *= d;
  }

  Tensor output("", input.data_type, out_shape,
                std::vector<uint8_t>(static_cast<std::size_t>(out_total) * elem_size));

  // Collect the selected axis indices.
  std::vector<int64_t> sel_axis_indices;
  sel_axis_indices.reserve(static_cast<std::size_t>(selected_count));
  for (int64_t i = 0; i < axis_dim && static_cast<std::size_t>(i) < cond_len; ++i) {
    if (cond[static_cast<std::size_t>(i)]) {
      sel_axis_indices.push_back(i);
    }
  }

  // Iterate over all output elements and copy from input.
  // We use a multi-index approach.
  const int64_t out_total_cnt = (out_shape.empty() ? 0 : out_total);
  if (out_total_cnt == 0) {
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

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t s = 0; s < selected_count; ++s) {
      const int64_t src_axis = sel_axis_indices[static_cast<std::size_t>(s)];
      // Source flat offset for this (outer, axis_idx) block.
      const int64_t src_base = (o * axis_dim + src_axis) * inner;
      const int64_t dst_base = (o * selected_count + s) * inner;
      std::memcpy(output.data.data() + static_cast<std::size_t>(dst_base) * elem_size,
                  input.data.data() + static_cast<std::size_t>(src_base) * elem_size,
                  static_cast<std::size_t>(inner) * elem_size);
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
  EXT_ENFORCE_INVALID(output.data.size() == produced.data.size(),
                      "kernel::Compress: preallocated output buffer size mismatch.");
  if (!produced.data.empty()) {
    std::memcpy(output.data.data(), produced.data.data(), produced.data.size());
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
