// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Computes the determinant of a single M x M matrix in row-major layout using
// LU decomposition with partial pivoting. The matrix is modified in place.
float DeterminantInPlace(float *a, int64_t m) {
  float det = 1.0f;
  for (int64_t i = 0; i < m; ++i) {
    // Find pivot row.
    int64_t pivot = i;
    float max_val = std::fabs(a[i * m + i]);
    for (int64_t r = i + 1; r < m; ++r) {
      const float v = std::fabs(a[r * m + i]);
      if (v > max_val) {
        max_val = v;
        pivot = r;
      }
    }
    if (max_val == 0.0f) {
      return 0.0f;
    }
    if (pivot != i) {
      for (int64_t c = 0; c < m; ++c) {
        const float tmp = a[i * m + c];
        a[i * m + c] = a[pivot * m + c];
        a[pivot * m + c] = tmp;
      }
      det = -det;
    }
    const float diag = a[i * m + i];
    det *= diag;
    // Eliminate below the pivot.
    for (int64_t r = i + 1; r < m; ++r) {
      const float factor = a[r * m + i] / diag;
      if (factor != 0.0f) {
        for (int64_t c = i + 1; c < m; ++c) {
          a[r * m + c] -= factor * a[i * m + c];
        }
      }
    }
  }
  return det;
}

} // namespace

Tensor Det::operator()(const Tensor &x) const {
  EXT_ENFORCE_INVALID(x.shape.size() >= 2, "kernel::Det requires an input tensor of rank >= 2.");
  const int64_t m = x.shape[x.shape.size() - 1];
  const int64_t m2 = x.shape[x.shape.size() - 2];
  EXT_ENFORCE_INVALID(m == m2, "kernel::Det requires the inner-most 2 dimensions to be equal.");
  std::vector<int64_t> out_shape(x.shape.begin(), x.shape.end() - 2);
  int64_t batch = 1;
  for (int64_t d : out_shape)
    batch *= d;
  Tensor y("", DataType::FLOAT, out_shape,
           std::vector<uint8_t>(static_cast<size_t>(batch) * sizeof(float)));
  (*this)(x, y);
  return y;
}

void Det::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT, "kernel::Det only supports FLOAT tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::FLOAT,
                      "kernel::Det preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2, "kernel::Det requires an input tensor of rank >= 2.");
  const int64_t m = x.shape[x.shape.size() - 1];
  const int64_t m2 = x.shape[x.shape.size() - 2];
  EXT_ENFORCE_INVALID(m == m2, "kernel::Det requires the inner-most 2 dimensions to be equal.");

  std::vector<int64_t> expected_out_shape(x.shape.begin(), x.shape.end() - 2);
  EXT_ENFORCE_INVALID(output.shape == expected_out_shape,
                      "kernel::Det preallocated output shape must match the batch dimensions "
                      "of the input.");
  int64_t batch = 1;
  for (int64_t d : expected_out_shape)
    batch *= d;
  const size_t expected_bytes = static_cast<size_t>(batch) * sizeof(float);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::Det preallocated output buffer has unexpected size in bytes.");

  const float *px = x.AsFloat();
  float *py = output.AsFloat();
  const int64_t matrix_size = m * m;
  std::vector<float> work(static_cast<size_t>(matrix_size));
  for (int64_t b = 0; b < batch; ++b) {
    const float *src = px + b * matrix_size;
    for (int64_t i = 0; i < matrix_size; ++i)
      work[static_cast<size_t>(i)] = src[i];
    py[static_cast<size_t>(b)] = DeterminantInPlace(work.data(), m);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
