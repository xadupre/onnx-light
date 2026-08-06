// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

onnx_kernels::Shape ComputeFlattenOutputShape(const onnx_kernels::Shape &in_shape, int64_t axis) {
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

Tensor Flatten::operator()(const Tensor &input, int64_t axis, RuntimeContext *rt) const {
  const onnx_kernels::Shape out_shape = ComputeFlattenOutputShape(input.shape, axis);
  const size_t output_n_bytes = PackedByteSize(input.data_type, input.element_count());
  Tensor output =
      MakeOutputTensor(input.data_type, out_shape, output_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(input, axis, output);
  return output;
}

void Flatten::operator()(const Tensor &input, int64_t axis, Tensor &output) const {
  const onnx_kernels::Shape out_shape = ComputeFlattenOutputShape(input.shape, axis);
  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::Flatten: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Flatten: preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(output.size_bytes() == input.size_bytes(),
                      "kernel::Flatten: preallocated output byte-size mismatch.");
  if (input.size_bytes() > 0) {
    std::memcpy(output.mutable_bytes(), input.bytes(), input.size_bytes());
  }
}

void Flatten::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const int64_t axis = GetAttributeIntOrDefault(node, "axis", 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, axis, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
