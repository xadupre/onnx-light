// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Sequence SequenceErase::operator()(const Sequence &input_sequence, const Tensor *position) const {
  const int64_t n = static_cast<int64_t>(input_sequence.size());

  // Resolve the erase position (default: last element).
  int64_t idx;
  if (position == nullptr) {
    EXT_ENFORCE_INVALID(n > 0, "kernel::SequenceErase: cannot erase from an empty sequence.");
    idx = n - 1;
  } else {
    EXT_ENFORCE_INVALID(!position->data.empty() && position->shape.empty(),
                        "kernel::SequenceErase: 'position' must be a scalar tensor.");
    if (position->data_type == static_cast<int32_t>(DataType::INT32)) {
      idx = static_cast<int64_t>(*position->AsInt32());
    } else {
      EXT_ENFORCE_INVALID(position->data_type == static_cast<int32_t>(DataType::INT64),
                          "kernel::SequenceErase: 'position' must have data type INT32 or INT64.");
      idx = *position->AsInt64();
    }
    // Normalize negative index.
    if (idx < 0) {
      idx += n;
    }
    EXT_ENFORCE_INVALID(idx >= 0 && idx < n, "kernel::SequenceErase: position ",
                        std::to_string(idx), " is out of range for sequence of length ",
                        std::to_string(n), ".");
  }

  // Build output sequence omitting element at idx.
  Tensors out_values;
  out_values.reserve(static_cast<std::size_t>(n - 1));
  for (int64_t i = 0; i < n; ++i) {
    if (i != idx) {
      out_values.push_back(input_sequence.values[static_cast<std::size_t>(i)]);
    }
  }
  return Sequence(input_sequence.name, input_sequence.elem_type, std::move(out_values));
}

void SequenceErase::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  EXT_ENFORCE_INVALID(!(node.input_size() > 2), "RunNode: op '", node.op_type(),
                      "' expects 1 or 2 inputs, got ", node.input_size(), ".");
  RequireOutputCount(node, 1);
  const Sequence &input_sequence = GetInputSequence(node, 0, rt);
  const Tensor *position = GetOptionalInput(node, 1, rt.tensors());
  onnx_kernels::kernel::SequenceErase k(rt.kernel_ctx());
  SetOutputSequence(node, 0, k(input_sequence, position), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
