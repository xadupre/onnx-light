// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/float16_promote.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Returns ``axis`` normalized to the ``[0, rank)`` range.
int64_t NormalizeAxis(int64_t axis, int64_t rank) {
  if (axis < 0) {
    axis += rank;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis < rank,
                      "kernel::RMSNormalization: axis out of range for X's rank.");
  return axis;
}

// Validates that ``scale``'s shape is unidirectionally broadcastable to the
// normalized shape ``X.shape[axis:]``.
void CheckScaleBroadcast(const onnx_kernels::Shape &x_shape, int64_t axis,
                         const onnx_kernels::Shape &scale_shape) {
  const int64_t normalized_rank = static_cast<int64_t>(x_shape.size()) - axis;
  EXT_ENFORCE_INVALID(static_cast<int64_t>(scale_shape.size()) <= normalized_rank,
                      "kernel::RMSNormalization: scale rank cannot exceed normalized rank.");
  const int64_t offset = normalized_rank - static_cast<int64_t>(scale_shape.size());
  for (size_t i = 0; i < scale_shape.size(); ++i) {
    const int64_t x_dim = x_shape[static_cast<size_t>(axis + offset) + i];
    const int64_t s_dim = scale_shape[i];
    EXT_ENFORCE_INVALID(s_dim == x_dim || s_dim == 1,
                        "kernel::RMSNormalization: scale shape is not broadcastable to "
                        "X's normalized shape.");
  }
}

} // namespace

Tensor RMSNormalization::operator()(const Tensor &x, const Tensor &scale, int64_t axis,
                                    float epsilon, RuntimeContext *rt) const {
  // FLOAT16/BFLOAT16 are computed in float32 and demoted back, mirroring the
  // half-precision dispatch used by kernel::Conv and kernel::MatMul. This lets
  // half-precision language models (e.g. the tiny Llama-style decoder) run
  // their RMSNorm layers through this kernel.
  if (IsHalfPrecision(x.data_type)) {
    RuntimeContext scratch_rt(
        rt ? rt->kernel_ctx() : ctx_,
        RuntimeContextOptions{.allocator = rt ? rt->execution_allocator() : nullptr});
    RuntimeContext *compute_rt = rt ? &scratch_rt : nullptr;
    const Tensor x_f = PromoteToFloat32(x, compute_rt);
    const Tensor scale_f = PromoteToFloat32(scale, compute_rt);
    Tensor y = (*this)(x_f, scale_f, axis, epsilon, compute_rt);
    Tensor demoted = DemoteFromFloat32(y, x.data_type, compute_rt);
    Tensor out = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, demoted.size_bytes())
                    : MakeOutputTensor(x.data_type, x.shape, demoted.size_bytes(), nullptr);
    if (demoted.size_bytes() != 0) {
      std::memcpy(out.mutable_bytes(), demoted.bytes(), demoted.size_bytes());
    }
    return out;
  }
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RMSNormalization: X must be FLOAT.");
  const size_t out_n_bytes = x.size_bytes();
  Tensor out =
      rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::FLOAT), x.shape, out_n_bytes)
         : MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), x.shape, out_n_bytes, nullptr);
  (*this)(x, scale, out, axis, epsilon, rt);
  return out;
}

void RMSNormalization::operator()(const Tensor &x, const Tensor &scale, Tensor &output,
                                  int64_t axis, float epsilon, RuntimeContext *rt) const {
  if (IsHalfPrecision(x.data_type)) {
    EXT_ENFORCE_INVALID(output.data_type == x.data_type,
                        "kernel::RMSNormalization preallocated output must match the input dtype.");
    Tensor y = (*this)(x, scale, axis, epsilon);
    EXT_ENFORCE_INVALID(output.shape == y.shape,
                        "kernel::RMSNormalization: output must have the same shape as X.");
    EXT_ENFORCE_INVALID(output.size_bytes() == y.size_bytes(),
                        "kernel::RMSNormalization: output buffer has unexpected size.");
    std::memcpy(output.mutable_bytes(), y.bytes(), y.size_bytes());
    return;
  }
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RMSNormalization: X must be FLOAT.");
  EXT_ENFORCE_INVALID(scale.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RMSNormalization: scale must be FLOAT.");
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RMSNormalization: output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::RMSNormalization: output must have the same shape as X.");
  EXT_ENFORCE_INVALID(output.size_bytes() == x.size_bytes(),
                      "kernel::RMSNormalization: output buffer must have the same byte size as X.");

  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank >= 1, "kernel::RMSNormalization: X must have rank >= 1.");
  axis = NormalizeAxis(axis, rank);

  CheckScaleBroadcast(x.shape, axis, scale.shape);

  // Compute the size of the outer block (axes [0, axis)) and the normalized
  // block (axes [axis, rank)).
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer *= x.shape[static_cast<size_t>(i)];
  }
  int64_t norm_size = 1;
  for (int64_t i = axis; i < rank; ++i) {
    norm_size *= x.shape[static_cast<size_t>(i)];
  }

  // Resolve the broadcast: a normalized-shape coordinate
  // ``(c_0, ..., c_{normalized_rank-1})`` maps to the last ``scale_rank``
  // coordinates in ``scale``, with any ``scale`` dim equal to 1 contributing 0.
  const int64_t normalized_rank = rank - axis;
  const int64_t scale_rank = static_cast<int64_t>(scale.shape.size());
  const int64_t offset = normalized_rank - scale_rank;

  // Scratch buffers are drawn from the runtime allocator backing ``output``
  // (when it is allocator-backed), mirroring kernel::QuantizeLinear; they fall
  // back to inline storage otherwise. ``TemporaryTypedBuffer`` leaves its
  // storage uninitialized, so any scratch that is read before being written is
  // cleared explicitly below.
  RawBufferAllocator *allocator = rt ? rt->execution_allocator() : nullptr;

  // Pre-compute the per-element index into ``scale`` for every position in the
  // normalized block. A scalar ``scale`` (scale_rank == 0) broadcasts to index
  // 0 everywhere, so the zeroed buffer is left untouched. The count is
  // clamped to at least 1 to avoid a zero-size allocation when ``norm_size`` is
  // 0 (the index buffer is then never read).
  const std::size_t scale_index_count = static_cast<std::size_t>(norm_size > 0 ? norm_size : 1);
  detail::TemporaryTypedBuffer<int64_t> scale_index_buf(scale_index_count, allocator,
                                                        "kernel::RMSNormalization scale_index");
  int64_t *scale_index = scale_index_buf.data();
  std::memset(scale_index, 0, scale_index_count * sizeof(int64_t));

  if (norm_size > 0 && scale_rank > 0) {
    detail::TemporaryTypedBuffer<int64_t> scale_strides_buf(
        static_cast<std::size_t>(scale_rank), allocator, "kernel::RMSNormalization scale_strides");
    int64_t *scale_strides = scale_strides_buf.data();
    int64_t stride = 1;
    for (int64_t i = scale_rank - 1; i >= 0; --i) {
      const int64_t dim = scale.shape[static_cast<size_t>(i)];
      scale_strides[static_cast<size_t>(i)] = dim == 1 ? 0 : stride;
      stride *= dim;
    }

    // Walk through the normalized block coordinates in row-major order
    // and accumulate the scale index using ``scale_strides``.
    detail::TemporaryTypedBuffer<int64_t> coord_buf(static_cast<std::size_t>(normalized_rank),
                                                    allocator, "kernel::RMSNormalization coord");
    int64_t *coord = coord_buf.data();
    std::memset(coord, 0, static_cast<std::size_t>(normalized_rank) * sizeof(int64_t));
    for (int64_t flat = 0; flat < norm_size; ++flat) {
      int64_t si = 0;
      for (int64_t i = offset; i < normalized_rank; ++i) {
        si += coord[static_cast<size_t>(i)] * scale_strides[static_cast<size_t>(i - offset)];
      }
      scale_index[static_cast<size_t>(flat)] = si;

      // Increment ``coord`` in row-major order (last dim varies fastest).
      for (int64_t i = normalized_rank - 1; i >= 0; --i) {
        ++coord[static_cast<size_t>(i)];
        if (coord[static_cast<size_t>(i)] < x.shape[static_cast<size_t>(axis + i)]) {
          break;
        }
        coord[static_cast<size_t>(i)] = 0;
      }
    }
  }

  const float *px = x.AsFloat();
  const float *ps = scale.AsFloat();
  float *py = output.AsFloat();

  // For each outer position, compute the mean of squares over the normalized
  // axes, take the square root and divide ``X`` by it. Then multiply by the
  // broadcasted scale.
  for (int64_t o = 0; o < outer; ++o) {
    const int64_t base = o * norm_size;
    double sqsum = 0.0;
    for (int64_t i = 0; i < norm_size; ++i) {
      const double v = static_cast<double>(px[base + i]);
      sqsum += v * v;
    }
    const double mean = norm_size > 0 ? sqsum / static_cast<double>(norm_size) : 0.0;
    const float inv_rms = 1.0f / std::sqrt(static_cast<float>(mean) + epsilon);
    for (int64_t i = 0; i < norm_size; ++i) {
      py[base + i] = px[base + i] * inv_rms * ps[scale_index[static_cast<size_t>(i)]];
    }
  }
}

void RMSNormalization::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const Tensor &scale = GetInput(node, 1, rt.tensors());
  onnx_kernels::kernel::RMSNormalization k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, scale, GetNormAxis(node), GetEpsilon(node), &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
