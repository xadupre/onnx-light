// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

std::vector<int64_t> ResolvePermOrDefault(const std::vector<int64_t> &perm, std::size_t rank) {
  if (perm.empty()) {
    std::vector<int64_t> reversed(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      reversed[i] = static_cast<int64_t>(rank - 1 - i);
    }
    return reversed;
  }
  EXT_ENFORCE_INVALID(perm.size() == rank, "kernel::Transpose: perm length must match input rank.");
  std::vector<bool> seen(rank, false);
  for (int64_t p : perm) {
    EXT_ENFORCE_INVALID(p >= 0 && static_cast<std::size_t>(p) < rank,
                        "kernel::Transpose: perm has an out-of-range axis.");
    EXT_ENFORCE_INVALID(!seen[static_cast<std::size_t>(p)],
                        "kernel::Transpose: perm has duplicate axes.");
    seen[static_cast<std::size_t>(p)] = true;
  }
  return perm;
}

std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return {};
  }
  std::vector<int64_t> strides(shape.size(), 1);
  for (std::size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

} // namespace

Tensor Transpose::operator()(const Tensor &data, const std::vector<int64_t> &perm) const {
  const std::vector<int64_t> resolved_perm = ResolvePermOrDefault(perm, data.shape.size());
  std::vector<int64_t> out_shape(resolved_perm.size());
  for (std::size_t i = 0; i < resolved_perm.size(); ++i) {
    out_shape[i] = data.shape[static_cast<std::size_t>(resolved_perm[i])];
  }

  Tensor output("", data.data_type, out_shape,
                std::vector<uint8_t>(PackedByteSize(data.data_type, data.element_count())));
  (*this)(data, perm, output);
  return output;
}

void Transpose::operator()(const Tensor &data, const std::vector<int64_t> &perm,
                           Tensor &output) const {
  const std::vector<int64_t> resolved_perm = ResolvePermOrDefault(perm, data.shape.size());
  std::vector<int64_t> out_shape(resolved_perm.size());
  for (std::size_t i = 0; i < resolved_perm.size(); ++i) {
    out_shape[i] = data.shape[static_cast<std::size_t>(resolved_perm[i])];
  }

  EXT_ENFORCE_INVALID(output.data_type == data.data_type,
                      "kernel::Transpose: preallocated output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Transpose: preallocated output shape mismatch.");

  const std::size_t elem_size = ElementSize(data.data_type);
  const std::vector<int64_t> in_strides = ComputeStrides(data.shape);
  const std::vector<int64_t> out_strides = ComputeStrides(out_shape);
  const int64_t total = output.element_count();

  for (int64_t out_idx = 0; out_idx < total; ++out_idx) {
    int64_t remaining = out_idx;
    int64_t in_idx = 0;
    for (std::size_t i = 0; i < out_shape.size(); ++i) {
      const int64_t coord = out_strides.empty() ? 0 : remaining / out_strides[i];
      if (!out_strides.empty()) {
        remaining %= out_strides[i];
      }
      in_idx += coord * in_strides[static_cast<std::size_t>(resolved_perm[i])];
    }
    std::memcpy(output.data.data() + static_cast<std::size_t>(out_idx) * elem_size,
                data.data.data() + static_cast<std::size_t>(in_idx) * elem_size, elem_size);
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
