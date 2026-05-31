// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank, "kernel::Softmax: axis is out of range.");
  return resolved;
}

} // namespace

Tensor Softmax::operator()(const Tensor &x, int64_t axis) const {
  Tensor y("", DataType::FLOAT, x.shape,
           std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * sizeof(float)));
  (*this)(x, axis, y);
  return y;
}

void Softmax::operator()(const Tensor &x, int64_t axis, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT,
                      "kernel::Softmax only supports FLOAT tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::FLOAT,
                      "kernel::Softmax preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Softmax preallocated output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.data.data() != x.data.data(),
                      "kernel::Softmax does not support aliasing input/output buffers.");

  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(float);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::Softmax preallocated output buffer has unexpected size in bytes.");

  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, "kernel::Softmax: input rank must be >= 1.");
  const int64_t resolved_axis = ResolveAxis(axis, rank);

  int64_t outer = 1;
  for (int64_t d = 0; d < resolved_axis; ++d) {
    outer *= x.shape[static_cast<size_t>(d)];
  }
  const int64_t axis_dim = x.shape[static_cast<size_t>(resolved_axis)];
  int64_t inner = 1;
  for (int64_t d = resolved_axis + 1; d < rank; ++d) {
    inner *= x.shape[static_cast<size_t>(d)];
  }

  const float *px = x.AsFloat();
  float *py = output.AsFloat();
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      float max_v = -std::numeric_limits<float>::infinity();
      for (int64_t a = 0; a < axis_dim; ++a) {
        const int64_t offset = (o * axis_dim + a) * inner + i;
        max_v = std::max(max_v, px[static_cast<size_t>(offset)]);
      }
      float sum = 0.0f;
      for (int64_t a = 0; a < axis_dim; ++a) {
        const int64_t offset = (o * axis_dim + a) * inner + i;
        sum += std::exp(px[static_cast<size_t>(offset)] - max_v);
      }
      for (int64_t a = 0; a < axis_dim; ++a) {
        const int64_t offset = (o * axis_dim + a) * inner + i;
        py[static_cast<size_t>(offset)] = std::exp(px[static_cast<size_t>(offset)] - max_v) / sum;
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
