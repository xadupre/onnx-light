// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Sequence SequenceEmpty::operator()(int32_t dtype) const {
  // Per ONNX schema, the default element type when 'dtype' is unspecified
  // (i.e. UNDEFINED == 0) is FLOAT.
  const int32_t elem_type = (dtype == static_cast<int32_t>(DataType::UNDEFINED))
                                ? static_cast<int32_t>(DataType::FLOAT)
                                : dtype;
  return Sequence(std::string{}, elem_type, Tensors{});
}

void SequenceEmpty::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 0);
  RequireOutputCount(node, 1);
  const int64_t dtype = GetAttributeIntOrDefault(node, "dtype", 0);
  onnx_kernels::kernel::SequenceEmpty k(rt.kernel_ctx());
  SetOutputSequence(node, 0, k(static_cast<int32_t>(dtype)), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
