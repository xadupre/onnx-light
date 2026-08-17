// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor SequenceLength::operator()(const Sequence &input_sequence, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(input_sequence.size() <=
                          static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
                      "kernel::SequenceLength: input sequence length exceeds int64_t range.");
  const int64_t size = static_cast<int64_t>(input_sequence.size());
  if (rt == nullptr) {
    return Tensor::FromInt64("", {}, {size}, ctx_.allocator);
  }
  Tensor output = rt->MakeOutputTensor(0, DataType::INT64, {}, sizeof(int64_t));
  output.AsInt64()[0] = size;
  return output;
}

void SequenceLength::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Sequence &input_sequence = GetInputSequence(node, 0, rt);
  onnx_kernels::kernel::SequenceLength k(rt.kernel_ctx());
  SetOutput(node, 0, k(input_sequence, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
