// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor SequenceAt::operator()(const Sequence &input_sequence, const Tensor &position,
                              RuntimeContext *rt) const {
  const int64_t n = static_cast<int64_t>(input_sequence.size());
  EXT_ENFORCE_INVALID(n > 0, "kernel::SequenceAt: cannot index into an empty sequence.");
  EXT_ENFORCE_INVALID(position.size_bytes() > 0 && position.shape.empty(),
                      "kernel::SequenceAt: 'position' must be a scalar tensor.");

  int64_t idx;
  if (position.data_type == static_cast<int32_t>(DataType::INT32)) {
    idx = static_cast<int64_t>(*position.AsInt32());
  } else {
    EXT_ENFORCE_INVALID(position.data_type == static_cast<int32_t>(DataType::INT64),
                        "kernel::SequenceAt: 'position' must have data type INT32 or INT64.");
    idx = *position.AsInt64();
  }
  // Normalize negative index.
  if (idx < 0) {
    idx += n;
  }
  EXT_ENFORCE_INVALID(idx >= 0 && idx < n, "kernel::SequenceAt: position ", std::to_string(idx),
                      " is out of range for sequence of length ", std::to_string(n), ".");
  const Tensor &input = input_sequence.values[static_cast<std::size_t>(idx)];
  const std::size_t n_bytes =
      input.data_type == static_cast<int32_t>(DataType::STRING) ? 0 : input.size_bytes();
  Tensor output = rt ? rt->MakeOutputTensor(0, input.data_type, input.shape, n_bytes)
                     : MakeOutputTensor(input.data_type, input.shape, n_bytes, nullptr);
  if (input.data_type == static_cast<int32_t>(DataType::STRING)) {
    output.string_data = input.string_data;
  } else if (n_bytes > 0) {
    std::memcpy(output.mutable_bytes(), input.bytes(), n_bytes);
  }
  return output;
}

void SequenceAt::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Sequence &input_sequence = GetInputSequence(node, 0, rt);
  const Tensor &position = GetInput(node, 1, rt.tensors());
  onnx_kernels::kernel::SequenceAt k(rt.kernel_ctx());
  SetOutput(node, 0, k(input_sequence, position, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
