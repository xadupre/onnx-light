// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Reads the 1-D INT64 ``repeats`` input tensor and validates it.
std::vector<int64_t> ReadTileRepeatsInput(const Tensor &repeats, std::size_t input_rank) {
  EXT_ENFORCE_INVALID(repeats.data_type == DataType::INT64,
                      "kernel::Tile: 'repeats' input must be INT64.");
  EXT_ENFORCE_INVALID(repeats.shape.size() == 1,
                      "kernel::Tile: 'repeats' input must be a 1-D tensor.");
  const int64_t n = repeats.shape[0];
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(n) == input_rank,
                      "kernel::Tile: 'repeats' length must equal the rank of 'input'.");
  std::vector<int64_t> out(static_cast<std::size_t>(n));
  if (n > 0) {
    std::memcpy(out.data(), repeats.data.data(), static_cast<std::size_t>(n) * sizeof(int64_t));
  }
  for (int64_t r : out) {
    EXT_ENFORCE_INVALID(r >= 0, "kernel::Tile: 'repeats' values must be non-negative.");
  }
  return out;
}

std::vector<int64_t> ComputeTileOutputShape(const std::vector<int64_t> &in_shape,
                                            const std::vector<int64_t> &repeats) {
  std::vector<int64_t> out_shape(in_shape.size());
  for (std::size_t k = 0; k < in_shape.size(); ++k) {
    out_shape[k] = in_shape[k] * repeats[k];
  }
  return out_shape;
}

} // namespace

Tensor Tile::operator()(const Tensor &input, const Tensor &repeats) const {
  const std::vector<int64_t> reps = ReadTileRepeatsInput(repeats, input.shape.size());
  const std::vector<int64_t> out_shape = ComputeTileOutputShape(input.shape, reps);
  const std::size_t elem_size = ElementSize(input.data_type);
  int64_t total_elements = 1;
  for (int64_t d : out_shape) {
    total_elements *= d;
  }
  Tensor out("", input.data_type, out_shape,
             std::vector<uint8_t>(static_cast<std::size_t>(total_elements) * elem_size));
  (*this)(input, repeats, out);
  return out;
}

void Tile::operator()(const Tensor &input, const Tensor &repeats, Tensor &output) const {
  const std::vector<int64_t> reps = ReadTileRepeatsInput(repeats, input.shape.size());
  const std::vector<int64_t> out_shape = ComputeTileOutputShape(input.shape, reps);

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
  std::vector<int64_t> in_strides(rank, 0);
  if (rank > 0) {
    in_strides[rank - 1] = 1;
    for (std::size_t k = rank - 1; k > 0; --k) {
      in_strides[k - 1] = in_strides[k] * input.shape[k];
    }
  }
  // Pre-compute output row-major strides.
  std::vector<int64_t> out_strides(rank, 0);
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
    std::memcpy(output.data.data() + static_cast<std::size_t>(out_idx) * elem_size,
                input.data.data() + static_cast<std::size_t>(in_idx) * elem_size, elem_size);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
