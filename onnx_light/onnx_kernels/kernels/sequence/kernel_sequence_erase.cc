// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

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
    EXT_ENFORCE_INVALID(idx >= 0 && idx < n,
                        "kernel::SequenceErase: position " + std::to_string(idx) +
                            " is out of range for sequence of length " + std::to_string(n) + ".");
  }

  // Build output sequence omitting element at idx.
  std::vector<Tensor> out_values;
  out_values.reserve(static_cast<std::size_t>(n - 1));
  for (int64_t i = 0; i < n; ++i) {
    if (i != idx) {
      out_values.push_back(input_sequence.values[static_cast<std::size_t>(i)]);
    }
  }
  return Sequence(input_sequence.name, input_sequence.elem_type, std::move(out_values));
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
