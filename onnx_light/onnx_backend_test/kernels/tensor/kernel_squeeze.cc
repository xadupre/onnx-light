// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

std::vector<int64_t> ResolveAxes(const std::vector<int64_t> &axes, int64_t rank) {
  std::vector<int64_t> resolved;
  resolved.reserve(axes.size());
  for (int64_t axis : axes) {
    const int64_t adjusted = axis < 0 ? axis + rank : axis;
    EXT_ENFORCE_INVALID(adjusted >= 0 && adjusted < rank, "kernel::Squeeze: axis out of range.");
    resolved.push_back(adjusted);
  }
  std::sort(resolved.begin(), resolved.end());
  EXT_ENFORCE_INVALID(std::adjacent_find(resolved.begin(), resolved.end()) == resolved.end(),
                      "kernel::Squeeze: duplicate axes are not allowed.");
  return resolved;
}

std::vector<int64_t> ComputeSqueezedShape(const Tensor &data, const std::vector<int64_t> &axes) {
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  std::vector<int64_t> out_shape;

  if (axes.empty()) {
    for (int64_t d : data.shape) {
      if (d != 1) {
        out_shape.push_back(d);
      }
    }
    return out_shape;
  }

  const std::vector<int64_t> resolved_axes = ResolveAxes(axes, rank);
  size_t axis_index = 0;
  for (int64_t i = 0; i < rank; ++i) {
    if (axis_index < resolved_axes.size() && resolved_axes[axis_index] == i) {
      EXT_ENFORCE_INVALID(data.shape[static_cast<size_t>(i)] == 1,
                          "kernel::Squeeze: selected axis dimension must be 1.");
      ++axis_index;
      continue;
    }
    out_shape.push_back(data.shape[static_cast<size_t>(i)]);
  }
  return out_shape;
}

} // namespace

Tensor Squeeze::operator()(const Tensor &data, const std::vector<int64_t> &axes) const {
  const std::vector<int64_t> out_shape = ComputeSqueezedShape(data, axes);
  Tensor output = data;
  output.name.clear();
  output.shape = out_shape;
  return output;
}

void Squeeze::operator()(const Tensor &data, const std::vector<int64_t> &axes,
                         Tensor &output) const {
  const std::vector<int64_t> out_shape = ComputeSqueezedShape(data, axes);
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Squeeze: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Squeeze: preallocated output shape mismatch.");
  if (data.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data = data.string_data;
    return;
  }
  EXT_ENFORCE_INVALID(output.data.size() == data.data.size(),
                      "kernel::Squeeze: preallocated output byte-size mismatch.");
  output.data = data.data;
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
