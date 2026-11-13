// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/dequantize_packed.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor DequantizePacked::operator()(const Tensor &x, const std::string &format, const Shape &shape,
                                    int64_t output_dtype, RuntimeContext *rt) const {
  return DequantizePackedTensor(x, format, shape, output_dtype,
                                rt ? rt->AllocatorForOutput(0) : ctx_.allocator);
}

void DequantizePacked::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const auto *shape = FindAttribute(node, "shape");
  EXT_ENFORCE_INVALID(shape != nullptr && shape->type() == AttributeProto::INTS,
                      "DequantizePacked: shape must be an INTS attribute.");
  const Shape dims = GetAttributeShapeOrDefault(node, "shape", {});
  SetOutput(node, 0,
            (*this)(GetInput(node, 0, rt), GetRequiredAttributeString(node, "format"), dims,
                    GetAttributeIntOrDefault(node, "output_dtype", TensorProto::FLOAT), &rt),
            rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
