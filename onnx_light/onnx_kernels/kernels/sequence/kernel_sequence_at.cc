// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

Tensor SequenceAt::operator()(const Sequence &input_sequence, const Tensor &position) const {
  const int64_t n = static_cast<int64_t>(input_sequence.size());
  EXT_ENFORCE_INVALID(n > 0, "kernel::SequenceAt: cannot index into an empty sequence.");
  EXT_ENFORCE_INVALID(!position.data.empty() && position.shape.empty(),
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
  EXT_ENFORCE_INVALID(idx >= 0 && idx < n, "kernel::SequenceAt: position " + std::to_string(idx) +
                                               " is out of range for sequence of length " +
                                               std::to_string(n) + ".");
  return input_sequence.values[static_cast<std::size_t>(idx)];
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
