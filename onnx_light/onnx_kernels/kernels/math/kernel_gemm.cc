// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include "onnx_kernels/kernels/_helpers/float16_promote.h"

#include "onnx_kernels/runtime_context.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kGemmName = "kernel::Gemm";

/// Computes Y = alpha * op(A) * op(B) + beta * C into a pre-allocated
/// result buffer.  ``op(X)`` transposes X when the corresponding
/// ``trans`` flag is non-zero.
///
/// A has shape (M, K) when transA=0, (K, M) when transA≠0.
/// B has shape (K, N) when transB=0, (N, K) when transB≠0.
/// C (optional) is unidirectionally broadcastable to (M, N).
template <typename T>
void GemmCompute(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                 int64_t transA, int64_t transB, std::vector<T> &result) {
  const int64_t m = transA ? a.shape[1] : a.shape[0];
  const int64_t k = transA ? a.shape[0] : a.shape[1];
  const int64_t n = transB ? b.shape[0] : b.shape[1];

  result.assign(static_cast<std::size_t>(m * n), T{0});

  const T *pa = a.As<T>();
  const T *pb = b.As<T>();

  // Y = alpha * op(A) * op(B)
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      T sum = T{0};
      for (int64_t l = 0; l < k; ++l) {
        const T a_val = transA ? pa[l * m + i] : pa[i * k + l];
        const T b_val = transB ? pb[j * k + l] : pb[l * n + j];
        sum += a_val * b_val;
      }
      result[static_cast<std::size_t>(i * n + j)] = static_cast<T>(alpha) * sum;
    }
  }

  // Add beta * C (with broadcasting)
  if (c != nullptr && beta != 0.0f) {
    const T *pc = c->As<T>();
    const int64_t c_rank = static_cast<int64_t>(c->shape.size());
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        T c_val;
        if (c_rank == 0) {
          // Scalar broadcast.
          c_val = pc[0];
        } else if (c_rank == 1) {
          // Shape (N,) — broadcast across rows.  A length-1 vector is treated
          // as a scalar (unidirectional broadcast to (M, N)).
          c_val = (c->shape[0] == 1) ? pc[0] : pc[j];
        } else {
          // Shape (M, N) or (1, N) or (M, 1) or (1, 1).
          const int64_t c_rows = c->shape[0];
          const int64_t c_cols = c->shape[1];
          const int64_t ci = (c_rows == 1) ? 0 : i;
          const int64_t cj = (c_cols == 1) ? 0 : j;
          c_val = pc[ci * c_cols + cj];
        }
        result[static_cast<std::size_t>(i * n + j)] += static_cast<T>(beta) * c_val;
      }
    }
  }
}

/// Allocates and returns the output tensor Y = alpha * op(A) * op(B) + beta * C.
template <typename T>
Tensor GemmAlloc(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                 int64_t transA, int64_t transB) {
  const int64_t m = transA ? a.shape[1] : a.shape[0];
  const int64_t n = transB ? b.shape[0] : b.shape[1];
  std::vector<T> result;
  GemmCompute<T>(a, b, c, alpha, beta, transA, transB, result);
  return Tensor::From<T>("", {m, n}, result);
}

/// Computes Y = alpha * op(A) * op(B) + beta * C into a preallocated output tensor.
/// Validates that @p output has the correct dtype and shape before writing.
template <typename T>
void GemmInPlace(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                 int64_t transA, int64_t transB, Tensor &output) {
  const int64_t m = transA ? a.shape[1] : a.shape[0];
  const int64_t n = transB ? b.shape[0] : b.shape[1];
  EXT_ENFORCE_INVALID(output.data_type == a.data_type, kGemmName,
                      " preallocated output must have the same dtype as input A.");
  EXT_ENFORCE_INVALID(output.shape.size() == 2 && output.shape[0] == m && output.shape[1] == n,
                      kGemmName, " preallocated output shape must be [", std::to_string(m), ", ",
                      std::to_string(n), "].");
  std::vector<T> result;
  GemmCompute<T>(a, b, c, alpha, beta, transA, transB, result);
  std::memcpy(output.mutable_bytes(), result.data(), result.size() * sizeof(T));
}

constexpr const char *kSupportedGemmTypesMsg =
    " only supports FLOAT, DOUBLE, FLOAT16 and BFLOAT16 inputs.";

// Promotes a (possibly half-precision) Gemm input to FLOAT32, leaving FLOAT/
// DOUBLE tensors untouched. Used by the half-precision fast paths so the
// reference computation can run in float32 and then be demoted back.
Tensor PromoteGemmInput(const Tensor &t) { return PromoteToFloat32(t); }

} // namespace

Tensor Gemm::operator()(RuntimeContext *rt, const Tensor &a, const Tensor &b, const Tensor *c,
                        float alpha, float beta, int64_t transA, int64_t transB) const {
  switch (a.data_type) {
  case DataType::FLOAT:
    return GemmAlloc<float>(a, b, c, alpha, beta, transA, transB);
  case DataType::DOUBLE:
    return GemmAlloc<double>(a, b, c, alpha, beta, transA, transB);
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    EXT_ENFORCE_INVALID(b.data_type == a.data_type, kGemmName,
                        " inputs A and B must share the same dtype.");
    const Tensor a_f = PromoteGemmInput(a);
    const Tensor b_f = PromoteGemmInput(b);
    Tensor c_f;
    const Tensor *c_ptr = nullptr;
    if (c != nullptr) {
      EXT_ENFORCE_INVALID(c->data_type == a.data_type, kGemmName,
                          " input C must share dtype with A and B.");
      c_f = PromoteGemmInput(*c);
      c_ptr = &c_f;
    }
    Tensor y = GemmAlloc<float>(a_f, b_f, c_ptr, alpha, beta, transA, transB);
    return DemoteFromFloat32(y, a.data_type);
  }
  default:
    EXT_THROW_INVALID(kGemmName, ": unsupported data type ", a.data_type, kSupportedGemmTypesMsg);
  }
}

void Gemm::operator()(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                      int64_t transA, int64_t transB, Tensor &output) const {
  switch (a.data_type) {
  case DataType::FLOAT:
    return GemmInPlace<float>(a, b, c, alpha, beta, transA, transB, output);
  case DataType::DOUBLE:
    return GemmInPlace<double>(a, b, c, alpha, beta, transA, transB, output);
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    EXT_ENFORCE_INVALID(output.data_type == a.data_type, kGemmName,
                        " preallocated output must have the same dtype as input A.");
    Tensor y = (*this)(nullptr, a, b, c, alpha, beta, transA, transB);
    EXT_ENFORCE_INVALID(output.shape == y.shape, kGemmName,
                        " preallocated output has an invalid shape.");
    EXT_ENFORCE_INVALID(output.size_bytes() == y.size_bytes(), kGemmName,
                        " preallocated output buffer size does not match its shape.");
    std::memcpy(output.mutable_bytes(), y.bytes(), y.size_bytes());
    return;
  }
  default:
    EXT_THROW_INVALID(kGemmName, ": unsupported data type ", a.data_type, kSupportedGemmTypesMsg);
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
