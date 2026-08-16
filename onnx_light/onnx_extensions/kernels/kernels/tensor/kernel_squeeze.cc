// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

onnx_kernels::Shape ResolveAxes(const onnx_kernels::Shape &axes, int64_t rank) {
  onnx_kernels::Shape resolved;
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

onnx_kernels::Shape ComputeSqueezedShape(const Tensor &data, const onnx_kernels::Shape &axes) {
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  onnx_kernels::Shape out_shape;

  if (axes.empty()) {
    for (int64_t d : data.shape) {
      if (d != 1) {
        out_shape.push_back(d);
      }
    }
    return out_shape;
  }

  const onnx_kernels::Shape resolved_axes = ResolveAxes(axes, rank);
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

Tensor Squeeze::operator()(const Tensor &data, const onnx_kernels::Shape &axes,
                           RuntimeContext * /*rt*/) const {
  const onnx_kernels::Shape out_shape = ComputeSqueezedShape(data, axes);
  Tensor output = data;
  output.name.clear();
  output.shape = out_shape;
  return output;
}

void Squeeze::operator()(const Tensor &data, const onnx_kernels::Shape &axes,
                         Tensor &output) const {
  const onnx_kernels::Shape out_shape = ComputeSqueezedShape(data, axes);
  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Squeeze: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Squeeze: preallocated output shape mismatch.");
  if (data.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data = data.string_data;
    return;
  }
  EXT_ENFORCE_INVALID(output.size_bytes() == data.size_bytes(),
                      "kernel::Squeeze: preallocated output byte-size mismatch.");
  std::memcpy(output.mutable_bytes(), data.bytes(), data.size_bytes());
}

void Squeeze::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  EXT_ENFORCE_INVALID(!(node.input_size() > 2), "RunNode: op 'Squeeze' expects at most 2 inputs.");
  RequireOutputCount(node, 1);
  const onnx_kernels::Shape axes_attr = GetAttributeIntsOrDefault(node, "axes", {});
  const Tensor &data = GetInput(node, 0, rt.tensors());
  onnx_kernels::Shape axes;
  const Tensor *axes_input = GetOptionalInput(node, 1, rt.tensors());
  if (axes_input != nullptr) {
    EXT_ENFORCE_INVALID(!(axes_input->data_type != static_cast<int32_t>(DataType::INT64) ||
                          axes_input->shape.size() > 1),
                        "RunNode: Squeeze 'axes' input must be a 1-D INT64 tensor.");
    const int64_t n = axes_input->element_count();
    const int64_t *p = axes_input->AsInt64();
    axes.assign(p, p + n);
  } else {
    axes = axes_attr;
  }
  SetOutput(node, 0, (*this)(data, axes, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
