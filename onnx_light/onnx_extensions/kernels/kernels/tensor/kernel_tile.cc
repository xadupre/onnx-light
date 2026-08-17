// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Reads and validates the 1-D INT64 ``repeats`` input tensor and copies its
// values into an allocator-backed :cpp:class:`Tensor` (obtained from
// :cpp:func:`MakeOutputTensor`) instead of a temporary ``std::vector<int64_t>``,
// so the scratch storage can come from the runtime allocator when one is given.
// Returns the INT64 ``Tensor`` holding the validated ``repeats`` values.
Tensor ReadTileRepeatsInput(const Tensor &repeats, std::size_t input_rank,
                            RawBufferAllocator *allocator) {
  EXT_ENFORCE_INVALID(repeats.data_type == DataType::INT64,
                      "kernel::Tile: 'repeats' input must be INT64.");
  EXT_ENFORCE_INVALID(repeats.shape.size() == 1,
                      "kernel::Tile: 'repeats' input must be a 1-D tensor.");
  const int64_t n = repeats.shape[0];
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(n) == input_rank,
                      "kernel::Tile: 'repeats' length must equal the rank of 'input'.");
  Tensor out = MakeOutputTensor(DataType::INT64, {n}, static_cast<std::size_t>(n) * sizeof(int64_t),
                                allocator);
  if (n > 0) {
    std::memcpy(out.mutable_bytes(), repeats.bytes(),
                static_cast<std::size_t>(n) * sizeof(int64_t));
  }
  const int64_t *reps = out.As<int64_t>();
  for (int64_t k = 0; k < n; ++k) {
    EXT_ENFORCE_INVALID(reps[k] >= 0, "kernel::Tile: 'repeats' values must be non-negative.");
  }
  return out;
}

onnx_kernels::Shape ComputeTileOutputShape(const onnx_kernels::Shape &in_shape,
                                           const int64_t *repeats) {
  onnx_kernels::Shape out_shape;
  out_shape.assign(in_shape.size(), 0);
  for (std::size_t k = 0; k < in_shape.size(); ++k) {
    out_shape[k] = in_shape[k] * repeats[k];
  }
  return out_shape;
}

} // namespace

Tensor Tile::operator()(const Tensor &input, const Tensor &repeats, RuntimeContext *rt) const {
  RawBufferAllocator *allocator = rt ? rt->execution_allocator() : nullptr;
  const Tensor reps = ReadTileRepeatsInput(repeats, input.shape.size(), allocator);
  const onnx_kernels::Shape out_shape = ComputeTileOutputShape(input.shape, reps.As<int64_t>());
  const std::size_t elem_size = ElementSize(input.data_type);
  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }
  const size_t out_n_bytes = static_cast<std::size_t>(total_elements) * elem_size;
  Tensor out = rt ? rt->MakeOutputTensor(0, input.data_type, out_shape, out_n_bytes)
                  : MakeOutputTensor(input.data_type, out_shape, out_n_bytes, nullptr);
  (*this)(input, repeats, out);
  return out;
}

void Tile::operator()(const Tensor &input, const Tensor &repeats, Tensor &output) const {
  const Tensor reps = ReadTileRepeatsInput(repeats, input.shape.size(), nullptr);
  const onnx_kernels::Shape out_shape = ComputeTileOutputShape(input.shape, reps.As<int64_t>());

  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::Tile: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Tile: preallocated output shape must match tiled shape.");

  const std::size_t elem_size = ElementSize(input.data_type);
  const std::size_t rank = out_shape.size();

  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }

  // Pre-compute input row-major strides (in elements).
  onnx_kernels::Shape in_strides;
  in_strides.assign(rank, 0);
  if (rank > 0) {
    in_strides[rank - 1] = 1;
    for (std::size_t k = rank - 1; k > 0; --k) {
      in_strides[k - 1] = in_strides[k] * input.shape[k];
    }
  }
  // Pre-compute output row-major strides.
  onnx_kernels::Shape out_strides;
  out_strides.assign(rank, 0);
  if (rank > 0) {
    out_strides[rank - 1] = 1;
    for (std::size_t k = rank - 1; k > 0; --k) {
      out_strides[k - 1] = out_strides[k] * out_shape[k];
    }
  }

  // For each output element, map back to the corresponding input element
  // by taking each coordinate modulo the input dimension on that axis.
  for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
    int64_t in_idx = 0;
    int64_t remaining = out_idx;
    for (std::size_t k = 0; k < rank; ++k) {
      const int64_t out_coord = remaining / out_strides[k];
      remaining %= out_strides[k];
      const int64_t in_coord = input.shape[k] == 0 ? 0 : (out_coord % input.shape[k]);
      in_idx += in_coord * in_strides[k];
    }
    std::memcpy(output.mutable_bytes() + static_cast<std::size_t>(out_idx) * elem_size,
                input.bytes() + static_cast<std::size_t>(in_idx) * elem_size, elem_size);
  }
}

void Tile::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &input = GetInput(node, 0, rt.tensors());
  const Tensor &repeats = GetInput(node, 1, rt.tensors());
  onnx_kernels::kernel::Tile k(rt.kernel_ctx());
  SetOutput(node, 0, k(input, repeats, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
