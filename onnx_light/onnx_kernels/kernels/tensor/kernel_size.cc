// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Returns the product of ``shape``'s entries, i.e. the total number of
// elements of a tensor with that shape. A 0-D tensor (empty ``shape``) has
// exactly one element.
int64_t ComputeNumElements(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    n *= d;
  }
  return n;
}

} // namespace

Tensor Size::operator()(const Tensor &data) const {
  const int64_t n = ComputeNumElements(data.shape);
  return Tensor::FromInt64("", {}, {n});
}

void Size::operator()(const Tensor &data, Tensor &output) const {
  const int64_t n = ComputeNumElements(data.shape);
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::Size: preallocated output dtype must be INT64.");
  EXT_ENFORCE_INVALID(output.shape == std::vector<int64_t>{},
                      "kernel::Size: preallocated output shape must be scalar.");
  EXT_ENFORCE_INVALID(output.data.size() == sizeof(int64_t),
                      "kernel::Size: preallocated output byte-size mismatch.");
  *reinterpret_cast<int64_t *>(output.data.data()) = n;
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
