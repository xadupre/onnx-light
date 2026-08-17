// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

onnx_kernels::Shape ResolveAxes(const onnx_kernels::Shape &axes, int64_t output_rank) {
  onnx_kernels::Shape resolved;
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

onnx_kernels::Shape ComputeUnsqueezedShape(const Tensor &data, const onnx_kernels::Shape &axes) {
  const int64_t input_rank = static_cast<int64_t>(data.shape.size());
  const int64_t output_rank = input_rank + static_cast<int64_t>(axes.size());
  const onnx_kernels::Shape resolved_axes = ResolveAxes(axes, output_rank);

  onnx_kernels::Shape out_shape;
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

Tensor Unsqueeze::operator()(const Tensor &data, const onnx_kernels::Shape &axes,
                             RuntimeContext *rt) const {
  const onnx_kernels::Shape out_shape = ComputeUnsqueezedShape(data, axes);
  Tensor output = (rt ? rt->MakeOutputTensor(0, data.data_type, out_shape, data.size_bytes())
                      : MakeOutputTensor(data.data_type, out_shape, data.size_bytes(), nullptr));
  output.name.clear();
  if (data.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data = data.string_data;
  } else if (data.size_bytes() != 0) {
    std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
  }
  return output;
}

void Unsqueeze::operator()(const Tensor &data, const onnx_kernels::Shape &axes,
                           Tensor &output) const {
  const onnx_kernels::Shape out_shape = ComputeUnsqueezedShape(data, axes);
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Unsqueeze: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Unsqueeze: preallocated output shape mismatch.");
  if (data.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data = data.string_data;
    return;
  }
  EXT_ENFORCE_INVALID(output.size_bytes() == data.size_bytes(),
                      "kernel::Unsqueeze: preallocated output byte-size mismatch.");
  std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
}

void Unsqueeze::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  EXT_ENFORCE_INVALID(!(node.input_size() > 2),
                      "RunNode: op 'Unsqueeze' expects at most 2 inputs.");
  RequireOutputCount(node, 1);
  const onnx_kernels::Shape axes_attr = GetAttributeIntsOrDefault(node, "axes", {});
  const Tensor &data = GetInput(node, 0, rt.tensors());
  onnx_kernels::Shape axes;
  const Tensor *axes_input = GetOptionalInput(node, 1, rt.tensors());
  if (axes_input != nullptr) {
    EXT_ENFORCE_INVALID(!(axes_input->data_type != static_cast<int32_t>(DataType::INT64) ||
                          axes_input->shape.size() > 1),
                        "RunNode: Unsqueeze 'axes' input must be a 1-D INT64 tensor.");
    const int64_t n = axes_input->element_count();
    const int64_t *p = axes_input->AsInt64();
    axes.assign(p, p + n);
  } else {
    axes = axes_attr;
  }
  SetOutput(node, 0, (*this)(data, axes, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
