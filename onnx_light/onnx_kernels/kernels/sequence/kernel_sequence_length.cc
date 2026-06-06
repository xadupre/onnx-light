// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor SequenceLength::operator()(const Sequence &input_sequence) const {
  EXT_ENFORCE_INVALID(input_sequence.size() <=
                          static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
                      "kernel::SequenceLength: input sequence length exceeds int64_t range.");
  return Tensor::FromInt64("", {}, {static_cast<int64_t>(input_sequence.size())});
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
