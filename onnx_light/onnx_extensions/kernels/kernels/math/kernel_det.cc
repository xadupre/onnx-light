// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Computes the determinant of a single M x M matrix in row-major layout using
// LU decomposition with partial pivoting. The matrix is modified in place.
template <typename T> T DeterminantInPlace(T *a, int64_t m) {
  T det = 1;
  for (int64_t i = 0; i < m; ++i) {
    // Find pivot row.
    int64_t pivot = i;
    T max_val = std::fabs(a[i * m + i]);
    for (int64_t r = i + 1; r < m; ++r) {
      const T v = std::fabs(a[r * m + i]);
      if (v > max_val) {
        max_val = v;
        pivot = r;
      }
    }
    if (max_val == 0.0f) {
      return 0;
    }
    if (pivot != i) {
      for (int64_t c = 0; c < m; ++c) {
        const T tmp = a[i * m + c];
        a[i * m + c] = a[pivot * m + c];
        a[pivot * m + c] = tmp;
      }
      det = -det;
    }
    const T diag = a[i * m + i];
    det *= diag;
    // Eliminate below the pivot.
    for (int64_t r = i + 1; r < m; ++r) {
      const T factor = a[r * m + i] / diag;
      if (factor != 0.0f) {
        for (int64_t c = i + 1; c < m; ++c) {
          a[r * m + c] -= factor * a[i * m + c];
        }
      }
    }
  }
  return det;
}

template <typename T, typename Stored, typename Decode, typename Encode>
void ComputeDet(const Tensor &x, Tensor &output, int64_t batch, int64_t m,
                RawBufferAllocator *allocator, Decode decode, Encode encode) {
  const Stored *px = reinterpret_cast<const Stored *>(x.bytes());
  Stored *py = reinterpret_cast<Stored *>(output.mutable_bytes());
  const int64_t matrix_size = m * m;
  detail::TemporaryTypedBuffer<T> work(static_cast<std::size_t>(matrix_size), allocator,
                                       "kernel::Det work");
  T *work_data = work.data();
  for (int64_t b = 0; b < batch; ++b) {
    const Stored *src = px + b * matrix_size;
    for (int64_t i = 0; i < matrix_size; ++i)
      work_data[static_cast<size_t>(i)] = decode(src[i]);
    py[static_cast<size_t>(b)] = encode(DeterminantInPlace(work_data, m));
  }
}

} // namespace

Tensor Det::operator()(const Tensor &x, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT || x.data_type == DataType::DOUBLE ||
                          x.data_type == DataType::FLOAT16 || x.data_type == DataType::BFLOAT16,
                      "kernel::Det only supports FLOAT, DOUBLE, FLOAT16, and BFLOAT16 tensors.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2, "kernel::Det requires an input tensor of rank >= 2.");
  const int64_t m = x.shape[x.shape.size() - 1];
  const int64_t m2 = x.shape[x.shape.size() - 2];
  EXT_ENFORCE_INVALID(m == m2, "kernel::Det requires the inner-most 2 dimensions to be equal.");
  Shape out_shape;
  out_shape.insert(out_shape.begin(), x.shape.begin(), x.shape.end() - 2);
  int64_t batch = 1;
  for (int64_t d : out_shape)
    batch *= d;
  const size_t y_n_bytes = static_cast<size_t>(batch) * x.element_size();
  Tensor y = rt ? rt->MakeOutputTensor(0, x.data_type, out_shape, y_n_bytes)
                : MakeOutputTensor(x.data_type, out_shape, y_n_bytes, nullptr);
  (*this)(x, y, rt ? rt->execution_allocator() : nullptr);
  return y;
}

void Det::operator()(const Tensor &x, Tensor &output,
                     RawBufferAllocator *temporary_allocator) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT || x.data_type == DataType::DOUBLE ||
                          x.data_type == DataType::FLOAT16 || x.data_type == DataType::BFLOAT16,
                      "kernel::Det only supports FLOAT, DOUBLE, FLOAT16, and BFLOAT16 tensors.");
  EXT_ENFORCE_INVALID(output.data_type == x.data_type,
                      "kernel::Det preallocated output must match the input type.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2, "kernel::Det requires an input tensor of rank >= 2.");
  const int64_t m = x.shape[x.shape.size() - 1];
  const int64_t m2 = x.shape[x.shape.size() - 2];
  EXT_ENFORCE_INVALID(m == m2, "kernel::Det requires the inner-most 2 dimensions to be equal.");

  Shape expected_out_shape;
  expected_out_shape.insert(expected_out_shape.begin(), x.shape.begin(), x.shape.end() - 2);
  EXT_ENFORCE_INVALID(output.shape == expected_out_shape,
                      "kernel::Det preallocated output shape must match the batch dimensions "
                      "of the input.");
  int64_t batch = 1;
  for (int64_t d : expected_out_shape)
    batch *= d;
  const size_t expected_bytes = static_cast<size_t>(batch) * x.element_size();
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes,
                      "kernel::Det preallocated output buffer has unexpected size in bytes.");

  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    ComputeDet<float, float>(
        x, output, batch, m, temporary_allocator, [](float v) { return v; },
        [](float v) { return v; });
    break;
  case DataType::DOUBLE:
    ComputeDet<double, double>(
        x, output, batch, m, temporary_allocator, [](double v) { return v; },
        [](double v) { return v; });
    break;
  case DataType::FLOAT16:
    ComputeDet<float, uint16_t>(x, output, batch, m, temporary_allocator, Float16BitsToFloat,
                                FloatToFloat16Bits);
    break;
  case DataType::BFLOAT16:
    ComputeDet<float, uint16_t>(x, output, batch, m, temporary_allocator, Bfloat16BitsToFloat,
                                FloatToBfloat16Bits);
    break;
  default:
    break;
  }
}

void Det::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
