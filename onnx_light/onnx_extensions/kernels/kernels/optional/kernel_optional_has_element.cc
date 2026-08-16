// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/optional/include_optional_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Materializes a scalar ``Tensor<bool, {}>`` carrying the given value, backed
// by the runtime context allocator when one is attached.
Tensor MakeScalarBool(bool value, RuntimeContext *rt) {
  const size_t out_n_bytes = 1;
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::BOOL), onnx_kernels::Shape{},
                                out_n_bytes, rt ? rt->allocator() : nullptr);
  *out.mutable_bytes() = value ? uint8_t{1} : uint8_t{0};
  return out;
}

} // namespace

Tensor OptionalHasElement::operator()(const Tensor &input, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(input.data_type != 0,
                      "kernel::OptionalHasElement: input element type must be a defined DataType.");
  // The runtime Tensor type cannot represent an "empty optional", so any
  // concrete tensor input is treated as the present element and the
  // operator returns true.
  return MakeScalarBool(true, rt);
}

Tensor OptionalHasElement::operator()(const Sequence &input, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(
      input.elem_type != 0,
      "kernel::OptionalHasElement: input sequence elem_type must be a defined DataType.");
  return MakeScalarBool(true, rt);
}

Tensor OptionalHasElement::operator()(RuntimeContext *rt) const {
  // Opset 18: an omitted input is reported as empty.
  return MakeScalarBool(false, rt);
}

void OptionalHasElement::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() > 1),
                      "RunNode: op 'OptionalHasElement' expects 0 or 1 inputs, got ",
                      node.input_size(), ".");
  RequireOutputCount(node, 1);
  onnx_kernels::kernel::OptionalHasElement k(rt.kernel_ctx());
  if (node.input_size() == 0 || node.input(0).empty()) {
    // Opset 18 omitted-input flavour: scalar ``false``.
    SetOutput(node, 0, k(&rt), rt);
    return;
  }
  const std::string input_name = node.input(0);
  if (rt.HasSequence(input_name)) {
    const Sequence &input_seq = GetInputSequence(node, 0, rt);
    SetOutput(node, 0, k(input_seq, &rt), rt);
  } else {
    const Tensor &input = GetInput(node, 0, rt.tensors());
    SetOutput(node, 0, k(input, &rt), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
