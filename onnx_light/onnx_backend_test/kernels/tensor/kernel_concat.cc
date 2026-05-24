// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor Concat::operator()(const std::vector<Tensor> &inputs, int64_t axis) const {
  if (inputs.empty()) {
    throw std::invalid_argument("kernel::Concat requires at least one input tensor.");
  }

  const int32_t dtype = inputs[0].data_type;
  const std::vector<int64_t> &shape0 = inputs[0].shape;
  const int64_t rank = static_cast<int64_t>(shape0.size());
  if (rank == 0) {
    throw std::invalid_argument("kernel::Concat cannot concatenate scalar tensors.");
  }

  // Resolve negative axis (ONNX semantics: axis in [-rank, rank-1]).
  int64_t resolved_axis = axis < 0 ? axis + rank : axis;
  if (resolved_axis < 0 || resolved_axis >= rank) {
    throw std::invalid_argument("kernel::Concat axis is out of range.");
  }

  // Validate shapes/dtypes and compute the output shape.
  std::vector<int64_t> out_shape = shape0;
  out_shape[static_cast<size_t>(resolved_axis)] = 0;
  for (const Tensor &t : inputs) {
    if (t.data_type != dtype) {
      throw std::invalid_argument("kernel::Concat requires all inputs to share the same dtype.");
    }
    if (static_cast<int64_t>(t.shape.size()) != rank) {
      throw std::invalid_argument("kernel::Concat requires all inputs to share the same rank.");
    }
    for (int64_t d = 0; d < rank; ++d) {
      if (d == resolved_axis) {
        continue;
      }
      if (t.shape[static_cast<size_t>(d)] != shape0[static_cast<size_t>(d)]) {
        throw std::invalid_argument(
            "kernel::Concat requires inputs to match on all non-axis dimensions.");
      }
    }
    out_shape[static_cast<size_t>(resolved_axis)] += t.shape[static_cast<size_t>(resolved_axis)];
  }

  // Outer is the product of dimensions before ``axis``; inner_bytes is the
  // product of dimensions after ``axis`` multiplied by the element size.
  int64_t outer = 1;
  for (int64_t d = 0; d < resolved_axis; ++d) {
    outer *= shape0[static_cast<size_t>(d)];
  }
  const size_t elem_size = ElementSize(dtype);
  int64_t inner = 1;
  for (int64_t d = resolved_axis + 1; d < rank; ++d) {
    inner *= shape0[static_cast<size_t>(d)];
  }
  const size_t inner_bytes = static_cast<size_t>(inner) * elem_size;
  const size_t out_axis = static_cast<size_t>(out_shape[static_cast<size_t>(resolved_axis)]);
  const size_t row_bytes = out_axis * inner_bytes;

  std::vector<uint8_t> out(static_cast<size_t>(outer) * row_bytes);
  size_t row_offset = 0;
  for (const Tensor &t : inputs) {
    const size_t axis_dim = static_cast<size_t>(t.shape[static_cast<size_t>(resolved_axis)]);
    const size_t block_bytes = axis_dim * inner_bytes;
    for (int64_t o = 0; o < outer; ++o) {
      std::memcpy(out.data() + static_cast<size_t>(o) * row_bytes + row_offset,
                  t.data.data() + static_cast<size_t>(o) * block_bytes, block_bytes);
    }
    row_offset += block_bytes;
  }

  return Tensor("", dtype, out_shape, std::move(out));
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
