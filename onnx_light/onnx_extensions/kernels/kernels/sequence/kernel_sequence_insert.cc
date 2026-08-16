// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Sequence SequenceInsert::operator()(const Sequence &input_sequence, const Tensor &tensor,
                                    const Tensor *position) const {
  const int64_t n = static_cast<int64_t>(input_sequence.size());

  const int32_t elem_type =
      input_sequence.elem_type == 0 ? tensor.data_type : input_sequence.elem_type;
  EXT_ENFORCE_INVALID(
      elem_type != 0,
      "kernel::SequenceInsert: element type must be defined for non-empty insertion.");
  EXT_ENFORCE_INVALID(
      tensor.data_type == elem_type,
      "kernel::SequenceInsert: inserted tensor dtype must match sequence element type.");

  int64_t idx = n;
  if (position != nullptr) {
    EXT_ENFORCE_INVALID(position->element_count() == 1,
                        "kernel::SequenceInsert: 'position' must be a single-element tensor "
                        "(a scalar or a tensor of shape [1]).");
    if (position->data_type == static_cast<int32_t>(DataType::INT32)) {
      idx = static_cast<int64_t>(*position->AsInt32());
    } else {
      EXT_ENFORCE_INVALID(position->data_type == static_cast<int32_t>(DataType::INT64),
                          "kernel::SequenceInsert: 'position' must have data type INT32 or INT64.");
      idx = *position->AsInt64();
    }
    if (idx < 0) {
      idx += n;
    }
    EXT_ENFORCE_INVALID(idx >= 0 && idx <= n, "kernel::SequenceInsert: position ",
                        std::to_string(idx), " is out of range for sequence of length ",
                        std::to_string(n), ".");
  }

  Tensors out_values = input_sequence.values;
  out_values.insert(out_values.begin() + idx, tensor);
  return Sequence(input_sequence.name, elem_type, std::move(out_values));
}

void SequenceInsert::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 2);
  EXT_ENFORCE_INVALID(!(node.input_size() > 3), "RunNode: op '", node.op_type(),
                      "' expects 2 or 3 inputs, got ", node.input_size(), ".");
  RequireOutputCount(node, 1);
  const Sequence &input_sequence = GetInputSequence(node, 0, rt);
  const Tensor &tensor = GetInput(node, 1, rt.tensors());
  const Tensor *position = GetOptionalInput(node, 2, rt.tensors());
  onnx_kernels::kernel::SequenceInsert k(rt.kernel_ctx());
  SetOutputSequence(node, 0, k(input_sequence, tensor, position), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
