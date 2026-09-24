// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include "onnx_extensions/kernels/kernels/quantization/include_quantization_kernels.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor Dequantize::operator()(const EncodedValueProto &x, int32_t dtype,
                              const StructTypeCatalogue &catalogue, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(dtype == TensorProto::FLOAT || dtype == TensorProto::DOUBLE ||
                          dtype == TensorProto::FLOAT16 || dtype == TensorProto::BFLOAT16,
                      "Dequantize dtype must be FLOAT, DOUBLE, FLOAT16 or BFLOAT16.");
  return DequantizeTensor(x, catalogue, rt ? rt->AllocatorForOutput(0) : ctx_.allocator, dtype);
}

void Dequantize::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const int64_t dtype = GetRequiredAttributeInt(node, "dtype");
  EXT_ENFORCE_INVALID(dtype == TensorProto::FLOAT || dtype == TensorProto::DOUBLE ||
                          dtype == TensorProto::FLOAT16 || dtype == TensorProto::BFLOAT16,
                      "Dequantize dtype must be FLOAT, DOUBLE, FLOAT16 or BFLOAT16.");
  const auto value = rt.values().find(node.input(0));
  EXT_ENFORCE_INVALID(value != rt.values().end() &&
                          value->second.kind == RuntimeValue::Kind::kEncoded,
                      "Dequantize requires an EncodedValueProto input.");
  SetOutput(node, 0,
            (*this)(value->second.Encoded(), static_cast<int32_t>(dtype),
                    rt.struct_type_catalogue(), &rt),
            rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
