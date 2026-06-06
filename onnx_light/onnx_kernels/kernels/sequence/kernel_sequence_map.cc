// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/sequence/include_sequence_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

std::vector<Sequence>
SequenceMap::operator()(const Sequence &input_sequence,
                        const std::vector<std::vector<Tensor>> &body_outputs_per_iter) const {
  const std::size_t n = input_sequence.size();

  std::vector<Sequence> outputs;
  outputs.reserve(body_outputs_per_iter.size());

  for (std::size_t k = 0; k < body_outputs_per_iter.size(); ++k) {
    const std::vector<Tensor> &row = body_outputs_per_iter[k];
    EXT_ENFORCE_INVALID(row.size() == n, "kernel::SequenceMap: body output #" + std::to_string(k) +
                                             " has " + std::to_string(row.size()) +
                                             " per-iteration tensors, expected " +
                                             std::to_string(n) +
                                             " (one per input sequence "
                                             "element).");

    // All per-iteration tensors of a given output must share a common
    // data type. The schema permits per-iteration shape variation, so
    // shapes are not required to match.
    int32_t elem_type = static_cast<int32_t>(DataType::UNDEFINED);
    if (!row.empty()) {
      elem_type = row.front().data_type;
      for (std::size_t i = 1; i < row.size(); ++i) {
        EXT_ENFORCE_INVALID(row[i].data_type == elem_type,
                            "kernel::SequenceMap: body output #" + std::to_string(k) +
                                " has tensors with mismatched data types across iterations.");
      }
    }

    outputs.emplace_back(Sequence("", elem_type, row));
  }

  return outputs;
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
