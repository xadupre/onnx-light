// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/math/matmul_shape_utils.h"

#include "onnx_core/runtime/kernels/float16_promote.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kMatMulName = "kernel::MatMul";
constexpr const char *kSupportedMatMulTypesMsg =
    " only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT32, INT64, UINT32 and UINT64 inputs.";

int64_t NumElements(const Shape &shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    n *= d;
  }
  return n;
}

Shape ComputeStrides(const Shape &shape) {
  Shape strides;
  strides.assign(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return strides;
}

Shape ComputeMatMulOutputShape(const Shape &a_shape, const Shape &b_shape) {
  return detail::ComputeMatMulOutputShape(
      a_shape, b_shape, kMatMulName, " does not accept rank-0 inputs.",
      " got incompatible inner dimensions.",
      " inputs are not broadcast-compatible on batch dimensions.");
}

template <typename T> void MatMulCompute(const Tensor &a, const Tensor &b, Tensor &output) {
  const Shape a2 = detail::PromoteMatMulShape(a.shape, true);
  const Shape b2 = detail::PromoteMatMulShape(b.shape, false);
  const int64_t m = a2[a2.size() - 2];
  const int64_t k = a2[a2.size() - 1];
  const int64_t n = b2[b2.size() - 1];
  EXT_ENFORCE_INVALID(k == b2[b2.size() - 2], kMatMulName, " got incompatible inner dimensions.");

  Shape a_prefix, b_prefix;
  a_prefix.insert(a_prefix.begin(), a2.begin(), a2.end() - 2);
  b_prefix.insert(b_prefix.begin(), b2.begin(), b2.end() - 2);
  const Shape out_prefix = detail::BroadcastMatMulPrefix(
      a_prefix, b_prefix, kMatMulName, " inputs are not broadcast-compatible on batch dimensions.");
  const size_t batch_rank = out_prefix.size();

  const Shape a_strides = ComputeStrides(a2);
  const Shape b_strides = ComputeStrides(b2);
  const Shape out_strides = ComputeStrides(output.shape);

  const T *pa = a.As<T>();
  const T *pb = b.As<T>();
  T *py = output.As<T>();

  const int64_t batch_count = NumElements(out_prefix);
  Shape batch_idx;
  batch_idx.assign(batch_rank, 0);
  const size_t a_prefix_rank = a_prefix.size();
  const size_t b_prefix_rank = b_prefix.size();

  // Strides addressing the (i, j) element of each output matrix. They are loop
  // invariant, so resolve the 1-D operand special cases once here rather than
  // per element. When A (resp. B) is 1-D the row (resp. column) dimension
  // collapses and its output stride is 0 (m, resp. n, is 1 in that case).
  const bool a_is_matrix = a.shape.size() != 1;
  const bool b_is_matrix = b.shape.size() != 1;
  int64_t y_row_stride = 0;
  int64_t y_col_stride = 0;
  if (a_is_matrix && b_is_matrix) {
    y_row_stride = out_strides[batch_rank];
    y_col_stride = out_strides[batch_rank + 1];
  } else if (!a_is_matrix && b_is_matrix) {
    y_col_stride = out_strides[batch_rank];
  } else if (a_is_matrix && !b_is_matrix) {
    y_row_stride = out_strides[batch_rank];
  }

  const int64_t a_row_stride = a_strides[a2.size() - 2];
  const int64_t a_k_stride = a_strides[a2.size() - 1];
  const int64_t b_k_stride = b_strides[b2.size() - 2];
  const int64_t b_col_stride = b_strides[b2.size() - 1];

  // The i-k-j loop below accumulates into the output, so start from zero. For
  // every output element it keeps the same k accumulation order as a plain
  // i-j-k dot product (k ascending), so the result stays bit-for-bit identical
  // while the innermost loop walks A/B/Y with contiguous (unit-stride) access
  // in the common row-major case, which the compiler can vectorise.
  std::memset(py, 0, output.size_bytes());

  for (int64_t batch = 0; batch < batch_count; ++batch) {
    int64_t a_base = 0;
    int64_t b_base = 0;
    int64_t y_base = 0;
    for (size_t d = 0; d < batch_rank; ++d) {
      const int64_t coord = batch_idx[d];
      if (d + a_prefix_rank >= batch_rank) {
        const size_t a_dim = d - (batch_rank - a_prefix_rank);
        const int64_t a_coord = (a_prefix[a_dim] == 1) ? 0 : coord;
        a_base += a_coord * a_strides[a_dim];
      }
      if (d + b_prefix_rank >= batch_rank) {
        const size_t b_dim = d - (batch_rank - b_prefix_rank);
        const int64_t b_coord = (b_prefix[b_dim] == 1) ? 0 : coord;
        b_base += b_coord * b_strides[b_dim];
      }
      y_base += coord * out_strides[d];
    }

    for (int64_t i = 0; i < m; ++i) {
      const int64_t a_row = a_base + i * a_row_stride;
      const int64_t y_row = y_base + i * y_row_stride;
      for (int64_t kk = 0; kk < k; ++kk) {
        const T av = pa[a_row + kk * a_k_stride];
        const int64_t b_row = b_base + kk * b_k_stride;
        for (int64_t j = 0; j < n; ++j) {
          py[y_row + j * y_col_stride] += av * pb[b_row + j * b_col_stride];
        }
      }
    }

    for (size_t d = batch_rank; d-- > 0;) {
      if (++batch_idx[d] < out_prefix[d]) {
        break;
      }
      batch_idx[d] = 0;
    }
  }
}

template <typename T>
Tensor MatMulAlloc(const Tensor &a, const Tensor &b, RawBufferAllocator *allocator = nullptr) {
  const Shape out_shape = ComputeMatMulOutputShape(a.shape, b.shape);
  const size_t y_n_bytes = static_cast<size_t>(NumElements(out_shape)) * sizeof(T);
  Tensor y = MakeOutputTensor(a.data_type, out_shape, y_n_bytes, allocator);
  MatMulCompute<T>(a, b, y);
  return y;
}

template <typename T> void MatMulInPlace(const Tensor &a, const Tensor &b, Tensor &output) {
  const Shape out_shape = ComputeMatMulOutputShape(a.shape, b.shape);
  EXT_ENFORCE_INVALID(output.data_type == a.data_type, kMatMulName,
                      " preallocated output must have the same dtype as input A.");
  EXT_ENFORCE_INVALID(output.shape == out_shape, kMatMulName,
                      " preallocated output has an invalid shape.");
  const size_t expected_bytes = static_cast<size_t>(NumElements(out_shape)) * sizeof(T);
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes, kMatMulName,
                      " preallocated output buffer size does not match its shape.");
  MatMulCompute<T>(a, b, output);
}

} // namespace

Tensor MatMul::operator()(const Tensor &a, const Tensor &b, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(a.data_type == b.data_type, kMatMulName,
                      " inputs must share the same dtype.");
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  switch (a.data_type) {
  case DataType::FLOAT:
    return MatMulAlloc<float>(a, b, allocator);
  case DataType::DOUBLE:
    return MatMulAlloc<double>(a, b, allocator);
  case DataType::INT32:
    return MatMulAlloc<int32_t>(a, b, allocator);
  case DataType::INT64:
    return MatMulAlloc<int64_t>(a, b, allocator);
  case DataType::UINT32:
    return MatMulAlloc<uint32_t>(a, b, allocator);
  case DataType::UINT64:
    return MatMulAlloc<uint64_t>(a, b, allocator);
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    const Tensor a_f = PromoteToFloat32(a, rt);
    const Tensor b_f = PromoteToFloat32(b, rt);
    Tensor y = MatMulAlloc<float>(a_f, b_f, allocator);
    return DemoteFromFloat32(y, a.data_type, rt);
  }
  default:
    EXT_THROW_INVALID(kMatMulName, ": unsupported data type ", a.data_type,
                      kSupportedMatMulTypesMsg);
  }
}

void MatMul::operator()(const Tensor &a, const Tensor &b, Tensor &output) const {
  EXT_ENFORCE_INVALID(a.data_type == b.data_type, kMatMulName,
                      " inputs must share the same dtype.");
  switch (a.data_type) {
  case DataType::FLOAT:
    return MatMulInPlace<float>(a, b, output);
  case DataType::DOUBLE:
    return MatMulInPlace<double>(a, b, output);
  case DataType::INT32:
    return MatMulInPlace<int32_t>(a, b, output);
  case DataType::INT64:
    return MatMulInPlace<int64_t>(a, b, output);
  case DataType::UINT32:
    return MatMulInPlace<uint32_t>(a, b, output);
  case DataType::UINT64:
    return MatMulInPlace<uint64_t>(a, b, output);
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    EXT_ENFORCE_INVALID(output.data_type == a.data_type, kMatMulName,
                        " preallocated output must have the same dtype as input A.");
    Tensor y = (*this)(a, b);
    EXT_ENFORCE_INVALID(output.shape == y.shape, kMatMulName,
                        " preallocated output has an invalid shape.");
    EXT_ENFORCE_INVALID(output.size_bytes() == y.size_bytes(), kMatMulName,
                        " preallocated output buffer size does not match its shape.");
    std::memcpy(output.mutable_bytes(), y.bytes(), y.size_bytes());
    return;
  }
  default:
    EXT_THROW_INVALID(kMatMulName, ": unsupported data type ", a.data_type,
                      kSupportedMatMulTypesMsg);
  }
}

void MatMul::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &y = GetInput(node, 1, rt.tensors());
  SetOutput(node, 0, (*this)(x, y, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
