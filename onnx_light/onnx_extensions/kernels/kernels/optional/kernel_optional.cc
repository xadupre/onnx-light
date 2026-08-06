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

Tensor Optional::operator()(const Tensor &input, RuntimeContext *rt) const {
  const size_t out_n_bytes = input.size_bytes();
  Tensor out =
      MakeOutputTensor(input.data_type, input.shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(input, out);
  return out;
}

void Optional::operator()(const Tensor &input, Tensor &output) const {
  EXT_ENFORCE_INVALID(input.data_type != 0,
                      "kernel::Optional: input element type must be a defined DataType.");
  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::Optional preallocated output data_type must match input data_type.");
  EXT_ENFORCE_INVALID(output.shape == input.shape,
                      "kernel::Optional preallocated output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == input.size_bytes(),
                      "kernel::Optional preallocated output buffer has unexpected size in bytes.");
  // Passthrough: the present optional wraps an exact copy of the input.
  // ``std::memmove``-style safety is required so the in-place overload may
  // alias ``input`` and ``output``.
  if (output.size_bytes() != 0 && output.mutable_bytes() != input.bytes()) {
    std::memcpy(output.mutable_bytes(), input.bytes(), input.size_bytes());
  }
}

void Optional::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const std::string input_name = node.input(0);
  if (rt.HasSequence(input_name)) {
    // Sequence-typed input: passthrough into the optional-of-sequence
    // output. The Optional kernel itself has no sequence overload
    // because the runtime ``Sequence`` already models the value, so
    // we copy the input sequence into the output slot directly.
    SetOutputSequence(node, 0, rt.GetSequence(input_name), rt);
  } else {
    const Tensor &input = GetInput(node, 0, rt.tensors());
    onnx_kernels::kernel::Optional k(rt.kernel_ctx());
    SetOutput(node, 0, k(input, &rt), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
