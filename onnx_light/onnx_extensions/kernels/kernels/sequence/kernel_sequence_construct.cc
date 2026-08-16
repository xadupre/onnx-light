// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/sequence/include_sequence_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Computes the stacked output shape ``[N, *first_shape]`` and the expected
// total byte size for an ``N``-element sequence whose elements have shape
// ``first_shape`` and per-element byte size ``per_elem_bytes``. Validates
// that every entry in ``inputs`` shares the first entry's ``data_type`` and
// ``shape``; throws ``std::invalid_argument`` otherwise.
void ValidateInputsAndComputeShape(const Tensors &inputs, onnx_kernels::Shape &stacked_shape,
                                   size_t &total_bytes) {
  const int64_t n = static_cast<int64_t>(inputs.size());
  if (n == 0) {
    stacked_shape = {0};
    total_bytes = 0;
    return;
  }
  const Tensor &first = inputs[0];
  EXT_ENFORCE_INVALID(first.data_type != 0,
                      "kernel::SequenceConstruct: input element type must be a defined "
                      "DataType.");
  const size_t per_elem_bytes = first.size_bytes();
  for (size_t i = 1; i < inputs.size(); ++i) {
    EXT_ENFORCE_INVALID(inputs[i].data_type == first.data_type,
                        "kernel::SequenceConstruct: all inputs must share the same data_type.");
    EXT_ENFORCE_INVALID(inputs[i].shape == first.shape,
                        "kernel::SequenceConstruct: all inputs must share the same shape.");
    EXT_ENFORCE_INVALID(inputs[i].size_bytes() == per_elem_bytes,
                        "kernel::SequenceConstruct: all inputs must share the same byte size.");
  }
  stacked_shape.assign(0, 0);
  stacked_shape.reserve(first.shape.size() + 1);
  stacked_shape.push_back(n);
  stacked_shape.insert(stacked_shape.end(), first.shape.begin(), first.shape.end());
  total_bytes = per_elem_bytes * static_cast<size_t>(n);
}

} // namespace

Tensor SequenceConstruct::operator()(const Tensors &inputs, RuntimeContext *rt) const {
  onnx_kernels::Shape stacked_shape;
  size_t total_bytes = 0;
  ValidateInputsAndComputeShape(inputs, stacked_shape, total_bytes);
  const int32_t out_dtype = inputs.empty() ? 0 : inputs[0].data_type;
  const size_t out_n_bytes = total_bytes;
  Tensor out =
      MakeOutputTensor(out_dtype, stacked_shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(inputs, out);
  return out;
}

void SequenceConstruct::operator()(const Tensors &inputs, Tensor &output) const {
  onnx_kernels::Shape stacked_shape;
  size_t total_bytes = 0;
  ValidateInputsAndComputeShape(inputs, stacked_shape, total_bytes);
  const int32_t expected_dtype = inputs.empty() ? 0 : inputs[0].data_type;
  EXT_ENFORCE_INVALID(
      output.data_type == expected_dtype,
      "kernel::SequenceConstruct preallocated output data_type must match input data_type.");
  EXT_ENFORCE_INVALID(
      output.shape == stacked_shape,
      "kernel::SequenceConstruct preallocated output shape must be [N, *input_shape].");
  EXT_ENFORCE_INVALID(
      output.size_bytes() == total_bytes,
      "kernel::SequenceConstruct preallocated output buffer has unexpected size in bytes.");
  size_t offset = 0;
  for (const Tensor &in : inputs) {
    if (in.size_bytes() > 0) {
      std::memcpy(output.mutable_bytes() + offset, in.bytes(), in.size_bytes());
    }
    offset += in.size_bytes();
  }
}

Sequence SequenceConstruct::AsSequence(const Tensors &inputs) const {
  Sequence out;
  out.values = inputs;
  if (inputs.empty()) {
    out.elem_type = 0;
    return out;
  }
  const int32_t expected_dtype = inputs[0].data_type;
  EXT_ENFORCE_INVALID(expected_dtype != 0,
                      "kernel::SequenceConstruct::AsSequence: input element type must be a defined "
                      "DataType.");
  for (size_t i = 1; i < inputs.size(); ++i) {
    EXT_ENFORCE_INVALID(
        inputs[i].data_type == expected_dtype,
        "kernel::SequenceConstruct::AsSequence: all inputs must share the same data_type.");
  }
  out.elem_type = expected_dtype;
  return out;
}

void SequenceConstruct::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  RequireOutputCount(node, 1);
  Tensors inputs;
  inputs.reserve(node.input_size());
  for (int i = 0; i < node.input_size(); ++i) {
    inputs.push_back(GetInput(node, i, rt.tensors()));
  }
  onnx_kernels::kernel::SequenceConstruct k(rt.kernel_ctx());
  SetOutputSequence(node, 0, k.AsSequence(inputs), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
