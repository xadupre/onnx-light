// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor Scatter::operator()(const Tensor &data, const Tensor &indices, const Tensor &updates,
                           const Attributes &attrs, RuntimeContext *rt) const {
  Tensor out = MakeOutputTensor(data.data_type, data.shape, data.size_bytes(),
                                rt != nullptr ? rt->allocator() : nullptr);
  (*this)(data, indices, updates, attrs, out);
  return out;
}

void Scatter::operator()(const Tensor &data, const Tensor &indices, const Tensor &updates,
                         const Attributes &attrs, Tensor &output) const {
  // ``Scatter`` (opset 9, deprecated since opset 11) is semantically
  // equivalent to ``ScatterElements`` with ``reduction="none"``; delegate to
  // that kernel to avoid duplicating the reference implementation.
  const ScatterElements se{this->ctx_};
  ScatterElements::Attributes se_attrs;
  se_attrs.axis = attrs.axis;
  se_attrs.reduction = "none";
  se(data, indices, updates, se_attrs, output);
}

void Scatter::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 3);
  RequireOutputCount(node, 1);
  const Tensor &data = GetInput(node, 0, rt.tensors());
  const Tensor &indices = GetInput(node, 1, rt.tensors());
  const Tensor &updates = GetInput(node, 2, rt.tensors());
  onnx_kernels::kernel::Scatter::Attributes attrs;
  attrs.axis = GetAttributeIntOrDefault(node, "axis", 0);
  onnx_kernels::kernel::Scatter k(rt.kernel_ctx());
  SetOutput(node, 0, k(data, indices, updates, attrs, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
