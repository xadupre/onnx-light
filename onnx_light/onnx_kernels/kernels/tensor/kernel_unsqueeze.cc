// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

std::vector<int64_t> ResolveAxes(const std::vector<int64_t> &axes, int64_t output_rank) {
  std::vector<int64_t> resolved;
  resolved.reserve(axes.size());
  for (int64_t axis : axes) {
    const int64_t adjusted = axis < 0 ? axis + output_rank : axis;
    EXT_ENFORCE_INVALID(adjusted >= 0 && adjusted < output_rank,
                        "kernel::Unsqueeze: axis out of range.");
    resolved.push_back(adjusted);
  }
  std::sort(resolved.begin(), resolved.end());
  EXT_ENFORCE_INVALID(std::adjacent_find(resolved.begin(), resolved.end()) == resolved.end(),
                      "kernel::Unsqueeze: duplicate axes are not allowed.");
  return resolved;
}

std::vector<int64_t> ComputeUnsqueezedShape(const Tensor &data, const std::vector<int64_t> &axes) {
  const int64_t input_rank = static_cast<int64_t>(data.shape.size());
  const int64_t output_rank = input_rank + static_cast<int64_t>(axes.size());
  const std::vector<int64_t> resolved_axes = ResolveAxes(axes, output_rank);

  std::vector<int64_t> out_shape;
  out_shape.reserve(static_cast<size_t>(output_rank));
  size_t axis_index = 0;
  size_t input_index = 0;
  for (int64_t out_i = 0; out_i < output_rank; ++out_i) {
    if (axis_index < resolved_axes.size() && resolved_axes[axis_index] == out_i) {
      out_shape.push_back(1);
      ++axis_index;
    } else {
      out_shape.push_back(data.shape[input_index]);
      ++input_index;
    }
  }
  return out_shape;
}

} // namespace

Tensor Unsqueeze::operator()(const Tensor &data, const std::vector<int64_t> &axes) const {
  const std::vector<int64_t> out_shape = ComputeUnsqueezedShape(data, axes);
  Tensor output = data;
  output.name.clear();
  output.shape = out_shape;
  return output;
}

void Unsqueeze::operator()(const Tensor &data, const std::vector<int64_t> &axes,
                           Tensor &output) const {
  const std::vector<int64_t> out_shape = ComputeUnsqueezedShape(data, axes);
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Unsqueeze: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Unsqueeze: preallocated output shape mismatch.");
  if (data.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data = data.string_data;
    return;
  }
  EXT_ENFORCE_INVALID(output.data.size() == data.size_bytes(),
                      "kernel::Unsqueeze: preallocated output byte-size mismatch.");
  std::memcpy(output.data.data(), data.bytes(), data.size_bytes());
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
