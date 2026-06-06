// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Sequence SequenceEmpty::operator()(int32_t dtype) const {
  // Per ONNX schema, the default element type when 'dtype' is unspecified
  // (i.e. UNDEFINED == 0) is FLOAT.
  const int32_t elem_type = (dtype == static_cast<int32_t>(DataType::UNDEFINED))
                                ? static_cast<int32_t>(DataType::FLOAT)
                                : dtype;
  return Sequence(std::string{}, elem_type, std::vector<Tensor>{});
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
