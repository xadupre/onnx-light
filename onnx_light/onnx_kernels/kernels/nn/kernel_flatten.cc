// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

std::vector<int64_t> ComputeFlattenOutputShape(const std::vector<int64_t> &in_shape, int64_t axis) {
  const int64_t rank = static_cast<int64_t>(in_shape.size());
  int64_t resolved_axis = axis;
  if (resolved_axis < 0) {
    resolved_axis += rank;
  }
  EXT_ENFORCE_INVALID(resolved_axis >= 0 && resolved_axis <= rank,
                      "kernel::Flatten: 'axis' must be in [-r, r] where r is the input rank.");
  int64_t outer = 1;
  for (int64_t i = 0; i < resolved_axis; ++i) {
    outer *= in_shape[static_cast<size_t>(i)];
  }
  int64_t inner = 1;
  for (int64_t i = resolved_axis; i < rank; ++i) {
    inner *= in_shape[static_cast<size_t>(i)];
  }
  return {outer, inner};
}

} // namespace

Tensor Flatten::operator()(const Tensor &input, int64_t axis) const {
  const std::vector<int64_t> out_shape = ComputeFlattenOutputShape(input.shape, axis);
  Tensor output("", input.data_type, out_shape,
                std::vector<uint8_t>(PackedByteSize(input.data_type, input.element_count())));
  (*this)(input, axis, output);
  return output;
}

void Flatten::operator()(const Tensor &input, int64_t axis, Tensor &output) const {
  const std::vector<int64_t> out_shape = ComputeFlattenOutputShape(input.shape, axis);
  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::Flatten: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Flatten: preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(output.data.size() == input.size_bytes(),
                      "kernel::Flatten: preallocated output byte-size mismatch.");
  if (input.size_bytes() > 0) {
    std::memcpy(output.data.data(), input.bytes(), input.size_bytes());
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
