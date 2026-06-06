// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Resolves the ONNX ``Shape`` operator's ``start``/``end`` attributes against
// an input of rank ``rank``. Negative values count from the back; the
// returned indices are clamped to ``[0, rank]`` per the upstream spec.
void ResolveStartEnd(const Shape::Attributes &attrs, int64_t rank, int64_t &start, int64_t &end) {
  start = attrs.start;
  if (start < 0) {
    start += rank;
  }
  if (start < 0) {
    start = 0;
  }
  if (start > rank) {
    start = rank;
  }

  end = attrs.end.has_value() ? *attrs.end : rank;
  if (end < 0) {
    end += rank;
  }
  if (end < 0) {
    end = 0;
  }
  if (end > rank) {
    end = rank;
  }
}

std::vector<int64_t> ComputeShapeSlice(const Tensor &data, const Shape::Attributes &attrs) {
  const int64_t rank = static_cast<int64_t>(data.shape.size());
  int64_t start = 0;
  int64_t end = 0;
  ResolveStartEnd(attrs, rank, start, end);
  std::vector<int64_t> values;
  if (end > start) {
    values.reserve(static_cast<std::size_t>(end - start));
    for (int64_t i = start; i < end; ++i) {
      values.push_back(data.shape[static_cast<std::size_t>(i)]);
    }
  }
  return values;
}

} // namespace

Tensor Shape::operator()(const Tensor &data) const { return (*this)(data, Attributes{}); }

Tensor Shape::operator()(const Tensor &data, const Attributes &attrs) const {
  const std::vector<int64_t> values = ComputeShapeSlice(data, attrs);
  const std::vector<int64_t> out_shape{static_cast<int64_t>(values.size())};
  return Tensor::FromInt64("", out_shape, values);
}

void Shape::operator()(const Tensor &data, const Attributes &attrs, Tensor &output) const {
  const std::vector<int64_t> values = ComputeShapeSlice(data, attrs);
  const std::vector<int64_t> out_shape{static_cast<int64_t>(values.size())};
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::Shape: preallocated output dtype must be INT64.");
  EXT_ENFORCE_INVALID(output.shape == out_shape,
                      "kernel::Shape: preallocated output shape mismatch.");
  EXT_ENFORCE_INVALID(output.data.size() == values.size() * sizeof(int64_t),
                      "kernel::Shape: preallocated output byte-size mismatch.");
  if (!values.empty()) {
    std::copy(values.begin(), values.end(), reinterpret_cast<int64_t *>(output.data.data()));
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
