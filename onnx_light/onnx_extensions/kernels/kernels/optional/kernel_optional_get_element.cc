// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/optional/include_optional_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor OptionalGetElement::operator()(const Tensor &input, RuntimeContext *rt) const {
  const size_t out_n_bytes = input.size_bytes();
  Tensor out =
      MakeOutputTensor(input.data_type, input.shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(input, out);
  return out;
}

void OptionalGetElement::operator()(const Tensor &input, Tensor &output) const {
  EXT_ENFORCE_INVALID(input.data_type != 0,
                      "kernel::OptionalGetElement: input element type must be a defined DataType.");
  EXT_ENFORCE_INVALID(
      output.data_type == input.data_type,
      "kernel::OptionalGetElement preallocated output data_type must match input data_type.");
  EXT_ENFORCE_INVALID(
      output.shape == input.shape,
      "kernel::OptionalGetElement preallocated output shape must match input shape.");
  EXT_ENFORCE_INVALID(
      output.size_bytes() == input.size_bytes(),
      "kernel::OptionalGetElement preallocated output buffer has unexpected size in bytes.");
  // Passthrough: the "present" optional is unwrapped to an exact copy of
  // the input. ``std::memmove``-style safety is required so the in-place
  // overload may alias ``input`` and ``output``.
  if (output.size_bytes() != 0 && output.mutable_bytes() != input.bytes()) {
    std::memcpy(output.mutable_bytes(), input.bytes(), input.size_bytes());
  }
}

Sequence OptionalGetElement::operator()(const Sequence &input) const {
  EXT_ENFORCE_INVALID(
      input.elem_type != 0,
      "kernel::OptionalGetElement: input sequence elem_type must be a defined DataType.");
  // Passthrough: return a copy of the input sequence.
  return Sequence(input.name, input.elem_type, input.values);
}

void OptionalGetElement::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const std::string input_name = node.input(0);
  onnx_kernels::kernel::OptionalGetElement k(rt.kernel_ctx());
  if (rt.HasSequence(input_name)) {
    const Sequence &input_seq = GetInputSequence(node, 0, rt);
    SetOutputSequence(node, 0, k(input_seq), rt);
  } else {
    const Tensor &input = GetInput(node, 0, rt.tensors());
    SetOutput(node, 0, k(input, &rt), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
