// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/float16_promote.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kGemmName = "kernel::Gemm";
constexpr uint32_t kTuningAbi = 1;

constexpr std::array<int32_t, 4> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16)};

int64_t DivideRoundUp(int64_t value, int64_t divisor) {
  return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

int64_t SaturatingMultiply(int64_t left, int64_t right) {
  if (left == 0 || right == 0) {
    return 0;
  }
  if (left > std::numeric_limits<int64_t>::max() / right) {
    return std::numeric_limits<int64_t>::max();
  }
  return left * right;
}

template <typename T> std::vector<T> PackB(const T *b, int64_t k, int64_t n, int64_t transB) {
  std::vector<T> packed(static_cast<size_t>(k * n));
  for (int64_t j = 0; j < n; ++j) {
    for (int64_t l = 0; l < k; ++l) {
      packed[static_cast<size_t>(j * k + l)] = transB ? b[j * k + l] : b[l * n + j];
    }
  }
  return packed;
}

/// Computes Y = alpha * op(A) * op(B) + beta * C into a caller-provided
/// result buffer.  ``op(X)`` transposes X when the corresponding
/// ``trans`` flag is non-zero.  ``result`` must point to storage large
/// enough to hold ``M * N`` elements.
///
/// A has shape (M, K) when transA=0, (K, M) when transA≠0.
/// B has shape (K, N) when transB=0, (N, K) when transB≠0.
/// C (optional) is unidirectionally broadcastable to (M, N).
template <typename T>
void GemmCompute(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                 int64_t transA, int64_t transB, const tuning::GemmTuning &tuning, T *result) {
  const int64_t m = transA ? a.shape[1] : a.shape[0];
  const int64_t k = transA ? a.shape[0] : a.shape[1];
  const int64_t n = transB ? b.shape[0] : b.shape[1];

  const T *pa = a.As<T>();
  const T *pb = b.As<T>();
  const bool pack_b = b.element_count() >= tuning.pack_b_minimum_elements;
  const std::vector<T> packed_b = pack_b ? PackB(pb, k, n, transB) : std::vector<T>{};
  const T *pc = c != nullptr ? c->As<T>() : nullptr;
  const int64_t c_rank = c != nullptr ? static_cast<int64_t>(c->shape.size()) : 0;

  const int64_t tile_m = m <= tuning.skinny_m_limit ? int64_t{1} : tuning.tile_m;
  const int64_t m_tiles = DivideRoundUp(m, tile_m);
  const int64_t n_tiles = DivideRoundUp(n, tuning.tile_n);
  const int64_t task_count = SaturatingMultiply(m_tiles, n_tiles);
  const int64_t tile_fmas = SaturatingMultiply(SaturatingMultiply(tile_m, tuning.tile_n), k);
  const int64_t task_grain = std::max<int64_t>(
      1, DivideRoundUp(tuning.parallel_fmas_per_work_unit, std::max<int64_t>(1, tile_fmas)));

  auto compute_tiles = [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t i_begin = (task / n_tiles) * tile_m;
      const int64_t j_begin = (task % n_tiles) * tuning.tile_n;
      const int64_t i_end = i_begin + std::min(tile_m, m - i_begin);
      const int64_t j_end = j_begin + std::min(tuning.tile_n, n - j_begin);
      for (int64_t i = i_begin; i < i_end; ++i) {
        for (int64_t j = j_begin; j < j_end; ++j) {
          T sum = T{0};
          for (int64_t l_begin = 0; l_begin < k; l_begin += tuning.tile_k) {
            const int64_t l_end = l_begin + std::min(tuning.tile_k, k - l_begin);
            for (int64_t l = l_begin; l < l_end; ++l) {
              const T a_val = transA ? pa[l * m + i] : pa[i * k + l];
              const T b_val = pack_b ? packed_b[static_cast<size_t>(j * k + l)]
                                     : (transB ? pb[j * k + l] : pb[l * n + j]);
              sum += a_val * b_val;
            }
          }
          T value = static_cast<T>(alpha) * sum;
          if (pc != nullptr && beta != 0.0f) {
            T c_val;
            if (c_rank == 0) {
              c_val = pc[0];
            } else if (c_rank == 1) {
              c_val = (c->shape[0] == 1) ? pc[0] : pc[j];
            } else {
              const int64_t c_rows = c->shape[0];
              const int64_t c_cols = c->shape[1];
              const int64_t ci = (c_rows == 1) ? 0 : i;
              const int64_t cj = (c_cols == 1) ? 0 : j;
              c_val = pc[ci * c_cols + cj];
            }
            value += static_cast<T>(beta) * c_val;
          }
          result[static_cast<size_t>(i * n + j)] = value;
        }
      }
    }
  };

  if (DivideRoundUp(task_count, task_grain) < tuning.parallel_minimum_tasks) {
    compute_tiles(0, task_count);
  } else {
    ParallelFor(task_count, task_grain, compute_tiles);
  }
}

/// Allocates and returns the output tensor Y = alpha * op(A) * op(B) + beta * C.
template <typename T>
Tensor GemmAlloc(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                 int64_t transA, int64_t transB, const tuning::GemmTuning &tuning,
                 RawBufferAllocator *allocator = nullptr) {
  const int64_t m = transA ? a.shape[1] : a.shape[0];
  const int64_t n = transB ? b.shape[0] : b.shape[1];
  const std::size_t n_bytes = static_cast<std::size_t>(m * n) * sizeof(T);
  Tensor y = MakeOutputTensor(TensorElementType<T>::value, {m, n}, n_bytes, allocator);
  GemmCompute<T>(a, b, c, alpha, beta, transA, transB, tuning, y.As<T>());
  return y;
}

/// Computes Y = alpha * op(A) * op(B) + beta * C into a preallocated output tensor.
/// Validates that @p output has the correct dtype and shape before writing.
template <typename T>
void GemmInPlace(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                 int64_t transA, int64_t transB, const tuning::GemmTuning &tuning, Tensor &output) {
  const int64_t m = transA ? a.shape[1] : a.shape[0];
  const int64_t n = transB ? b.shape[0] : b.shape[1];
  EXT_ENFORCE_INVALID(output.data_type == a.data_type, kGemmName,
                      " preallocated output must have the same dtype as input A.");
  EXT_ENFORCE_INVALID(output.shape.size() == 2 && output.shape[0] == m && output.shape[1] == n,
                      kGemmName, " preallocated output shape must be [", std::to_string(m), ", ",
                      std::to_string(n), "].");
  GemmCompute<T>(a, b, c, alpha, beta, transA, transB, tuning, output.As<T>());
}

constexpr const char *kSupportedGemmTypesMsg =
    " only supports FLOAT, DOUBLE, FLOAT16 and BFLOAT16 inputs.";

} // namespace

Gemm::Gemm(const KernelContext &ctx) : KernelBase(ctx) {}

void Gemm::RegisterTuningSchemas() {
  tuning::RegisterGemmTuningSchemas(kSupportedElementTypes, kTuningAbi);
}

KernelTuningKey Gemm::TuningKey(int32_t element_type) const {
  return tuning::IsSupportedElementType(element_type, kSupportedElementTypes)
             ? tuning::MakePortableTuningKey("Gemm", element_type, kTuningAbi)
             : KernelTuningKey{};
}

void Gemm::Configure(const KernelTuningParameters &parameters) {
  tuning::ConfigureGemmTuning(parameters, tuning_, kTuningAbi);
}

Tensor Gemm::operator()(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                        int64_t transA, int64_t transB, RuntimeContext *rt) const {
  switch (a.data_type) {
  case DataType::FLOAT:
    return GemmAlloc<float>(a, b, c, alpha, beta, transA, transB, tuning_, ctx_.allocator);
  case DataType::DOUBLE:
    return GemmAlloc<double>(a, b, c, alpha, beta, transA, transB, tuning_, ctx_.allocator);
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    EXT_ENFORCE_INVALID(b.data_type == a.data_type, kGemmName,
                        " inputs A and B must share the same dtype.");
    const Tensor a_f = PromoteToFloat32(a, rt, tuning_.conversion_parallel_minimum_elements);
    const Tensor b_f = PromoteToFloat32(b, rt, tuning_.conversion_parallel_minimum_elements);
    Tensor c_f;
    const Tensor *c_ptr = nullptr;
    if (c != nullptr) {
      EXT_ENFORCE_INVALID(c->data_type == a.data_type, kGemmName,
                          " input C must share dtype with A and B.");
      c_f = PromoteToFloat32(*c, rt, tuning_.conversion_parallel_minimum_elements);
      c_ptr = &c_f;
    }
    Tensor y = GemmAlloc<float>(a_f, b_f, c_ptr, alpha, beta, transA, transB, tuning_);
    return DemoteFromFloat32(y, a.data_type, rt, tuning_.conversion_parallel_minimum_elements);
  }
  default:
    EXT_THROW_INVALID(kGemmName, ": unsupported data type ", a.data_type, kSupportedGemmTypesMsg);
  }
}

void Gemm::operator()(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                      int64_t transA, int64_t transB, Tensor &output) const {
  switch (a.data_type) {
  case DataType::FLOAT:
    return GemmInPlace<float>(a, b, c, alpha, beta, transA, transB, tuning_, output);
  case DataType::DOUBLE:
    return GemmInPlace<double>(a, b, c, alpha, beta, transA, transB, tuning_, output);
  case DataType::FLOAT16:
  case DataType::BFLOAT16: {
    EXT_ENFORCE_INVALID(output.data_type == a.data_type, kGemmName,
                        " preallocated output must have the same dtype as input A.");
    Tensor y = (*this)(a, b, c, alpha, beta, transA, transB);
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

void Gemm::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 2 || node.input_size() > 3), "RunNode: op '",
                      node.op_type(), "' expects between 2 and 3 input(s), got ", node.input_size(),
                      ".");
  RequireOutputCount(node, 1);
  const Tensor &a = GetInput(node, 0, rt.tensors());
  const Tensor &b = GetInput(node, 1, rt.tensors());
  const Tensor *c = GetOptionalInput(node, 2, rt.tensors());
  const float alpha = GetAttributeFloatOrDefault(node, "alpha", 1.0f);
  const float beta = GetAttributeFloatOrDefault(node, "beta", 1.0f);
  const int64_t transA = GetAttributeIntOrDefault(node, "transA", 0);
  const int64_t transB = GetAttributeIntOrDefault(node, "transB", 0);
  SetOutput(node, 0, (*this)(a, b, c, alpha, beta, transA, transB, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
