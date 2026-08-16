// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstring>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Shared validation helper. Verifies all inputs share dtype + rank and match
// on every non-axis dimension. Returns the resolved axis (always >= 0), the
// output shape and the element size in bytes of ``inputs[0].data_type``.
struct ConcatLayout {
  int64_t axis;
  onnx_kernels::Shape shape;
  size_t elem_size;
};

ConcatLayout ValidateAndComputeLayout(const Tensors &inputs, int64_t axis) {
  EXT_ENFORCE_INVALID(!inputs.empty(), "kernel::Concat requires at least one input tensor.");

  const int32_t dtype = inputs[0].data_type;
  const onnx_kernels::Shape &shape0 = inputs[0].shape;
  const int64_t rank = static_cast<int64_t>(shape0.size());
  EXT_ENFORCE_INVALID(rank != 0, "kernel::Concat cannot concatenate scalar tensors.");

  // Resolve negative axis (ONNX semantics: axis in [-rank, rank-1]).
  const int64_t resolved_axis = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved_axis >= 0 && resolved_axis < rank,
                      "kernel::Concat axis is out of range.");

  onnx_kernels::Shape out_shape = shape0;
  out_shape[static_cast<size_t>(resolved_axis)] = 0;
  for (const Tensor &t : inputs) {
    EXT_ENFORCE_INVALID(t.data_type == dtype,
                        "kernel::Concat requires all inputs to share the same dtype.");
    EXT_ENFORCE_INVALID(static_cast<int64_t>(t.shape.size()) == rank,
                        "kernel::Concat requires all inputs to share the same rank.");
    for (int64_t d = 0; d < rank; ++d) {
      if (d == resolved_axis) {
        continue;
      }
      EXT_ENFORCE_INVALID(t.shape[static_cast<size_t>(d)] == shape0[static_cast<size_t>(d)],
                          "kernel::Concat requires inputs to match on all non-axis dimensions.");
    }
    out_shape[static_cast<size_t>(resolved_axis)] += t.shape[static_cast<size_t>(resolved_axis)];
  }

  return {resolved_axis, std::move(out_shape), ElementSize(dtype)};
}

} // namespace

Tensor Concat::operator()(const Tensors &inputs, int64_t axis, RuntimeContext *rt) const {
  const ConcatLayout layout = ValidateAndComputeLayout(inputs, axis);
  int64_t total = 1;
  for (int64_t d : layout.shape) {
    total *= d;
  }
  const size_t out_n_bytes = static_cast<size_t>(total) * layout.elem_size;
  Tensor out = MakeOutputTensor(inputs[0].data_type, layout.shape, out_n_bytes,
                                rt ? rt->allocator() : nullptr);
  (*this)(inputs, axis, out);
  return out;
}

void Concat::operator()(const Tensors &inputs, int64_t axis, Tensor &output) const {
  const ConcatLayout layout = ValidateAndComputeLayout(inputs, axis);
  const int32_t dtype = inputs[0].data_type;
  EXT_ENFORCE_INVALID(output.data_type == dtype,
                      "kernel::Concat preallocated output dtype must match inputs.");
  EXT_ENFORCE_INVALID(
      output.shape == layout.shape,
      "kernel::Concat preallocated output shape must match the concatenated shape.");

  // Outer is the product of dimensions before ``axis``; inner_bytes is the
  // product of dimensions after ``axis`` multiplied by the element size.
  const int64_t resolved_axis = layout.axis;
  const onnx_kernels::Shape &shape0 = inputs[0].shape;
  const int64_t rank = static_cast<int64_t>(shape0.size());
  int64_t outer = 1;
  for (int64_t d = 0; d < resolved_axis; ++d) {
    outer *= shape0[static_cast<size_t>(d)];
  }
  int64_t inner = 1;
  for (int64_t d = resolved_axis + 1; d < rank; ++d) {
    inner *= shape0[static_cast<size_t>(d)];
  }
  const size_t inner_bytes = static_cast<size_t>(inner) * layout.elem_size;
  const size_t out_axis = static_cast<size_t>(layout.shape[static_cast<size_t>(resolved_axis)]);
  const size_t row_bytes = out_axis * inner_bytes;

  const size_t expected_bytes = static_cast<size_t>(outer) * row_bytes;
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes,
                      "kernel::Concat preallocated output buffer has unexpected size in bytes.");

  size_t row_offset = 0;
  for (const Tensor &t : inputs) {
    const size_t axis_dim = static_cast<size_t>(t.shape[static_cast<size_t>(resolved_axis)]);
    const size_t block_bytes = axis_dim * inner_bytes;
    for (int64_t o = 0; o < outer; ++o) {
      std::memcpy(output.mutable_bytes() + static_cast<size_t>(o) * row_bytes + row_offset,
                  t.bytes() + static_cast<size_t>(o) * block_bytes, block_bytes);
    }
    row_offset += block_bytes;
  }
}

void Concat::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireMinInputCount(node, 1);
  RequireOutputCount(node, 1);
  Tensors inputs;
  inputs.reserve(node.input_size());
  for (int i = 0; i < node.input_size(); ++i) {
    inputs.push_back(GetInput(node, i, rt.tensors()));
  }
  const AttributeProto *axis_attr = FindAttribute(node, "axis");
  EXT_ENFORCE_INVALID(axis_attr != nullptr,
                      "RunNode: op 'Concat' is missing required attribute 'axis'.");
  const int64_t axis = axis_attr->i();
  onnx_kernels::kernel::Concat k(rt.kernel_ctx());
  SetOutput(node, 0, k(inputs, axis, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
