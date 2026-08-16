// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor CastLike::operator()(const Tensor &x, const Tensor &target_type, RuntimeContext *rt) const {
  // ``CastLike`` is defined by the spec as ``Cast`` with the ``to`` attribute
  // taken from the second input's element type. The values of ``target_type``
  // are ignored — only its ``data_type`` is observed.
  const Cast cast_kernel{ctx_};
  return cast_kernel(x, target_type.data_type, rt);
}

Tensor CastLike::operator()(const Tensor &x, const Tensor &target_type, bool saturate,
                            RuntimeContext *rt) const {
  const Cast cast_kernel{ctx_};
  return cast_kernel(x, target_type.data_type, saturate, rt);
}

void CastLike::operator()(const Tensor &x, const Tensor &target_type, Tensor &output) const {
  const Cast cast_kernel{ctx_};
  cast_kernel(x, target_type.data_type, output);
}

void CastLike::operator()(const Tensor &x, const Tensor &target_type, bool saturate,
                          Tensor &output) const {
  const Cast cast_kernel{ctx_};
  cast_kernel(x, target_type.data_type, saturate, output);
}

void CastLike::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &target_type = GetInput(node, 1, rt.tensors());
  const bool saturate = GetAttributeIntOrDefault(node, "saturate", 1) != 0;
  onnx_kernels::kernel::CastLike k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, target_type, saturate, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
