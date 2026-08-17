// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor Constant::operator()(const Tensor &value, RuntimeContext *rt) const {
  static_cast<void>(rt);
  if (value.data_type == static_cast<int32_t>(DataType::STRING)) {
    return Tensor::BorrowStrings(value.name, value.shape, value.AsStrings());
  }
  return Tensor::Borrow(value.name, value.data_type, value.shape, value.bytes(),
                        value.size_bytes());
}

Tensor Constant::operator()(Tensor &&value, RuntimeContext *rt) const {
  static_cast<void>(rt);
  return std::move(value);
}

void Constant::operator()(const Tensor &value, Tensor &output) const {
  EXT_ENFORCE_INVALID(
      output.data_type == value.data_type,
      "kernel::Constant preallocated output must have the same data type as the value.");
  EXT_ENFORCE_INVALID(output.shape == value.shape,
                      "kernel::Constant preallocated output shape must match the value shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == value.size_bytes(),
                      "kernel::Constant preallocated output buffer has unexpected size in bytes.");
  if (value.size_bytes() > 0) {
    std::memcpy(output.mutable_bytes(), value.bytes(), value.size_bytes());
  }
}

void Constant::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireOutputCount(node, 1);
  onnx_kernels::kernel::Constant k(rt.kernel_ctx());
  Tensor y;
  if (FindAttribute(node, "value") != nullptr) {
    y = k(GetRequiredAttributeTensor(node, "value"));
  } else if (FindAttribute(node, "value_float") != nullptr) {
    const float v = GetAttributeFloatOrDefault(node, "value_float", 0.0f);
    y = Tensor::FromFloat("", /*shape=*/{}, {v});
  } else if (FindAttribute(node, "value_floats") != nullptr) {
    const std::vector<float> vs = GetAttributeFloatsOrDefault(node, "value_floats", {});
    y = Tensor::FromFloat("", {static_cast<int64_t>(vs.size())}, vs);
  } else if (FindAttribute(node, "value_int") != nullptr) {
    const int64_t v = GetAttributeIntOrDefault(node, "value_int", 0);
    y = Tensor::FromInt64("", /*shape=*/{}, {v});
  } else if (FindAttribute(node, "value_ints") != nullptr) {
    const std::vector<int64_t> vs = GetAttributeIntsOrDefault(node, "value_ints", {});
    y = Tensor::FromInt64("", {static_cast<int64_t>(vs.size())}, vs);
  } else if (FindAttribute(node, "value_string") != nullptr) {
    const std::string v = GetAttributeStringOrDefault(node, "value_string", "");
    y = Tensor::FromStrings("", /*shape=*/{}, {v});
  } else if (FindAttribute(node, "value_strings") != nullptr) {
    const std::vector<std::string> vs = GetAttributeStringsOrDefault(node, "value_strings", {});
    y = Tensor::FromStrings("", {static_cast<int64_t>(vs.size())}, vs);
  } else {
    EXT_THROW_INVALID("RunNode: op 'Constant' requires one of: value, value_float, "
                      "value_floats, value_int, value_ints, value_string, value_strings.");
  }
  if (y.is_borrowed()) {
    SetOutput(node, 0, std::move(y), rt.tensors());
    return;
  }
  Tensor output = rt.MakeOutputTensor(0, y.data_type, y.shape, y.size_bytes());
  output.name = y.name;
  if (y.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data = std::move(y.string_data);
  } else if (y.size_bytes() > 0) {
    std::memcpy(output.mutable_bytes(), y.bytes(), y.size_bytes());
  }
  SetOutput(node, 0, std::move(output), rt.tensors());
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
