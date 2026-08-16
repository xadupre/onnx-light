// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor Size::operator()(const Tensor &data, RuntimeContext * /*rt*/) const {
  const int64_t n = data.shape.product();
  return Tensor::FromInt64("", {}, {n}, ctx_.allocator);
}

void Size::operator()(const Tensor &data, Tensor &output) const {
  const int64_t n = data.shape.product();
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::Size: preallocated output dtype must be INT64.");
  EXT_ENFORCE_INVALID(output.shape == onnx_kernels::Shape{},
                      "kernel::Size: preallocated output shape must be scalar.");
  EXT_ENFORCE_INVALID(output.size_bytes() == sizeof(int64_t),
                      "kernel::Size: preallocated output byte-size mismatch.");
  *reinterpret_cast<int64_t *>(output.mutable_bytes()) = n;
}

void Size::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &data = GetInput(node, 0, rt.tensors());
  onnx_kernels::kernel::Size k(rt.kernel_ctx());
  SetOutput(node, 0, k(data, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
