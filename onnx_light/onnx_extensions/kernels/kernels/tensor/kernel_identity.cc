// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor Identity::operator()(const Tensor &input, RuntimeContext *rt) const {
  Tensor output =
      (rt ? rt->MakeOutputTensor(0, input.data_type, input.shape, input.size_bytes())
          : MakeOutputTensor(input.data_type, input.shape, input.size_bytes(), nullptr));
  output.name = input.name;
  if (input.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data = input.string_data;
  } else if (input.size_bytes() != 0) {
    std::memcpy(output.mutable_bytes(), input.bytes(), input.size_bytes());
  }
  return output;
}

void Identity::operator()(const Tensor &input, Tensor &output) const {
  EXT_ENFORCE_INVALID(output.data_type == input.data_type,
                      "kernel::Identity: preallocated output dtype mismatch.");
  EXT_ENFORCE_INVALID(output.shape == input.shape,
                      "kernel::Identity: preallocated output shape mismatch.");
  if (input.data_type == static_cast<int32_t>(DataType::STRING)) {
    EXT_ENFORCE_INVALID(output.string_data.size() == input.string_data.size(),
                        "kernel::Identity: preallocated string output size mismatch.");
    output.string_data = input.string_data;
  } else {
    EXT_ENFORCE_INVALID(output.size_bytes() == input.size_bytes(),
                        "kernel::Identity: preallocated output byte-size mismatch.");
    if (input.size_bytes() != 0) {
      std::memcpy(output.mutable_bytes(), input.bytes(), input.size_bytes());
    }
  }
}

void Identity::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const std::string input_name = node.input(0);
  if (rt.HasSequence(input_name)) {
    SetOutputSequence(node, 0, rt.GetSequence(input_name), rt);
  } else {
    const Tensor &x = GetInput(node, 0, rt.tensors());
    onnx_kernels::kernel::Identity k(rt.kernel_ctx());
    SetOutput(node, 0, k(x, &rt), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
