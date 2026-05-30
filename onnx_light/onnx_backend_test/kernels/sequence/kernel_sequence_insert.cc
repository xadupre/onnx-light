// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

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
    EXT_ENFORCE_INVALID(!position->data.empty() && position->shape.empty(),
                        "kernel::SequenceInsert: 'position' must be a scalar tensor.");
    if (position->data_type == static_cast<int32_t>(TensorProto::DataType::INT32)) {
      idx = static_cast<int64_t>(*position->AsInt32());
    } else {
      EXT_ENFORCE_INVALID(position->data_type == static_cast<int32_t>(TensorProto::DataType::INT64),
                          "kernel::SequenceInsert: 'position' must have data type INT32 or INT64.");
      idx = *position->AsInt64();
    }
    if (idx < 0) {
      idx += n;
    }
    EXT_ENFORCE_INVALID(idx >= 0 && idx <= n,
                        "kernel::SequenceInsert: position " + std::to_string(idx) +
                            " is out of range for sequence of length " + std::to_string(n) + ".");
  }

  std::vector<Tensor> out_values = input_sequence.values;
  out_values.insert(out_values.begin() + idx, tensor);
  return Sequence(input_sequence.name, elem_type, std::move(out_values));
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
