// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/sequence/include_sequence_kernels.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Computes the stacked output shape ``[N, *first_shape]`` and the expected
// total byte size for an ``N``-element sequence whose elements have shape
// ``first_shape`` and per-element byte size ``per_elem_bytes``. Validates
// that every entry in ``inputs`` shares the first entry's ``data_type`` and
// ``shape``; throws ``std::invalid_argument`` otherwise.
void ValidateInputsAndComputeShape(const std::vector<Tensor> &inputs,
                                   std::vector<int64_t> &stacked_shape, size_t &total_bytes) {
  const int64_t n = static_cast<int64_t>(inputs.size());
  if (n == 0) {
    stacked_shape = {0};
    total_bytes = 0;
    return;
  }
  const Tensor &first = inputs[0];
  if (first.data_type == 0) {
    throw std::invalid_argument("kernel::SequenceConstruct: input element type must be a defined "
                                "TensorProto::DataType.");
  }
  const size_t per_elem_bytes = first.data.size();
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i].data_type != first.data_type) {
      throw std::invalid_argument(
          "kernel::SequenceConstruct: all inputs must share the same data_type.");
    }
    if (inputs[i].shape != first.shape) {
      throw std::invalid_argument(
          "kernel::SequenceConstruct: all inputs must share the same shape.");
    }
    if (inputs[i].data.size() != per_elem_bytes) {
      throw std::invalid_argument(
          "kernel::SequenceConstruct: all inputs must share the same byte size.");
    }
  }
  stacked_shape.clear();
  stacked_shape.reserve(first.shape.size() + 1);
  stacked_shape.push_back(n);
  stacked_shape.insert(stacked_shape.end(), first.shape.begin(), first.shape.end());
  total_bytes = per_elem_bytes * static_cast<size_t>(n);
}

} // namespace

Tensor SequenceConstruct::operator()(const std::vector<Tensor> &inputs) const {
  std::vector<int64_t> stacked_shape;
  size_t total_bytes = 0;
  ValidateInputsAndComputeShape(inputs, stacked_shape, total_bytes);
  const int32_t out_dtype = inputs.empty() ? 0 : inputs[0].data_type;
  Tensor out("", out_dtype, stacked_shape, std::vector<uint8_t>(total_bytes));
  (*this)(inputs, out);
  return out;
}

void SequenceConstruct::operator()(const std::vector<Tensor> &inputs, Tensor &output) const {
  std::vector<int64_t> stacked_shape;
  size_t total_bytes = 0;
  ValidateInputsAndComputeShape(inputs, stacked_shape, total_bytes);
  const int32_t expected_dtype = inputs.empty() ? 0 : inputs[0].data_type;
  if (output.data_type != expected_dtype) {
    throw std::invalid_argument(
        "kernel::SequenceConstruct preallocated output data_type must match input data_type.");
  }
  if (output.shape != stacked_shape) {
    throw std::invalid_argument(
        "kernel::SequenceConstruct preallocated output shape must be [N, *input_shape].");
  }
  if (output.data.size() != total_bytes) {
    throw std::invalid_argument(
        "kernel::SequenceConstruct preallocated output buffer has unexpected size in bytes.");
  }
  size_t offset = 0;
  for (const Tensor &in : inputs) {
    if (!in.data.empty()) {
      std::copy(in.data.begin(), in.data.end(), output.data.begin() + offset);
    }
    offset += in.data.size();
  }
}

Sequence SequenceConstruct::AsSequence(const std::vector<Tensor> &inputs) const {
  Sequence out;
  out.values = inputs;
  if (inputs.empty()) {
    out.elem_type = 0;
    return out;
  }
  const int32_t expected_dtype = inputs[0].data_type;
  if (expected_dtype == 0) {
    throw std::invalid_argument(
        "kernel::SequenceConstruct::AsSequence: input element type must be a defined "
        "TensorProto::DataType.");
  }
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i].data_type != expected_dtype) {
      throw std::invalid_argument(
          "kernel::SequenceConstruct::AsSequence: all inputs must share the same data_type.");
    }
  }
  out.elem_type = expected_dtype;
  return out;
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
