// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

onnx_kernels::Shape ResolvePermOrDefault(const onnx_kernels::Shape &perm, std::size_t rank) {
  if (perm.empty()) {
    onnx_kernels::Shape reversed;
    reversed.assign(rank, 0);
    for (std::size_t i = 0; i < rank; ++i) {
      reversed[i] = static_cast<int64_t>(rank - 1 - i);
    }
    return reversed;
  }
  EXT_ENFORCE_INVALID(perm.size() == rank, "kernel::Transpose: perm length must match input rank.");
  onnx_kernels::Shape seen;
  seen.assign(rank, 0);
  for (int64_t p : perm) {
    EXT_ENFORCE_INVALID(p >= 0 && static_cast<std::size_t>(p) < rank,
                        "kernel::Transpose: perm has an out-of-range axis.");
    EXT_ENFORCE_INVALID(!seen[static_cast<std::size_t>(p)],
                        "kernel::Transpose: perm has duplicate axes.");
    seen[static_cast<std::size_t>(p)] = 1;
  }
  return perm;
}

onnx_kernels::Shape ComputeStrides(const onnx_kernels::Shape &shape) {
  if (shape.empty()) {
    return {};
  }
  onnx_kernels::Shape strides;
  strides.assign(shape.size(), 1);
  for (std::size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

} // namespace

Tensor Transpose::operator()(const Tensor &data, const onnx_kernels::Shape &perm,
                             RuntimeContext *rt) const {
  const onnx_kernels::Shape resolved_perm = ResolvePermOrDefault(perm, data.shape.size());
  onnx_kernels::Shape out_shape;
  out_shape.assign(resolved_perm.size(), 0);
  for (std::size_t i = 0; i < resolved_perm.size(); ++i) {
    out_shape[i] = data.shape[static_cast<std::size_t>(resolved_perm[i])];
  }

  const size_t output_n_bytes = PackedByteSize(data.data_type, data.element_count());
  Tensor output =
      MakeOutputTensor(data.data_type, out_shape, output_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(data, perm, output);
  return output;
}

void Transpose::operator()(const Tensor &data, const onnx_kernels::Shape &perm,
                           Tensor &output) const {
  const onnx_kernels::Shape resolved_perm = ResolvePermOrDefault(perm, data.shape.size());
  onnx_kernels::Shape out_shape;
  out_shape.assign(resolved_perm.size(), 0);
  for (std::size_t i = 0; i < resolved_perm.size(); ++i) {
    out_shape[i] = data.shape[static_cast<std::size_t>(resolved_perm[i])];
  }

  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Transpose: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Transpose: preallocated output shape mismatch.");

  const std::size_t elem_size = ElementSize(data.data_type);
  const onnx_kernels::Shape in_strides = ComputeStrides(data.shape);
  const onnx_kernels::Shape out_strides = ComputeStrides(out_shape);
  const int64_t total = output.element_count();

  for (int64_t out_idx = 0; out_idx < total; ++out_idx) {
    int64_t remaining = out_idx;
    int64_t in_idx = 0;
    for (std::size_t i = 0; i < out_shape.size(); ++i) {
      const int64_t coord = remaining / out_strides[i];
      remaining %= out_strides[i];
      in_idx += coord * in_strides[static_cast<std::size_t>(resolved_perm[i])];
    }
    std::memcpy(output.mutable_bytes() + static_cast<std::size_t>(out_idx) * elem_size,
                data.bytes() + static_cast<std::size_t>(in_idx) * elem_size, elem_size);
  }
}

void Transpose::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &data = GetInput(node, 0, rt.tensors());
  const onnx_kernels::Shape perm = GetAttributeIntsOrDefault(node, "perm", {});
  onnx_kernels::kernel::Transpose k(rt.kernel_ctx());
  SetOutput(node, 0, k(data, perm, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
