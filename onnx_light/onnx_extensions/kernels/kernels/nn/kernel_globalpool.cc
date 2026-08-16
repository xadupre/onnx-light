// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Returns the total number of spatial elements per (n, c) slice.
int64_t SpatialCount(const Tensor &x) {
  int64_t count = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) {
    count *= x.shape[i];
  }
  return count;
}

} // namespace

// ---------------------------------------------------------------------------
// GlobalAveragePool
// ---------------------------------------------------------------------------

Tensor GlobalAveragePool::operator()(const Tensor &x, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GlobalAveragePool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2,
                      "kernel::GlobalAveragePool: x must have rank >= 2 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(SpatialCount(x) > 0,
                      "kernel::GlobalAveragePool: spatial extent must be non-zero.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t spatial = SpatialCount(x);

  // Build output shape: (N, C, 1, 1, ..., 1).
  onnx_kernels::Shape out_shape;
  out_shape.assign(x.shape.size(), 1);
  out_shape[0] = N;
  out_shape[1] = C;

  const size_t out_n_bytes = static_cast<size_t>(N * C) * sizeof(float);
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes,
                                rt ? rt->allocator() : nullptr);

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.mutable_bytes());

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t base = (n * C + c) * spatial;
      double sum = 0.0;
      for (int64_t s = 0; s < spatial; ++s) {
        sum += static_cast<double>(px[base + s]);
      }
      py[n * C + c] = static_cast<float>(sum / static_cast<double>(spatial));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// GlobalMaxPool
// ---------------------------------------------------------------------------

Tensor GlobalMaxPool::operator()(const Tensor &x, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GlobalMaxPool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2,
                      "kernel::GlobalMaxPool: x must have rank >= 2 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(SpatialCount(x) > 0,
                      "kernel::GlobalMaxPool: spatial extent must be non-zero.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t spatial = SpatialCount(x);

  onnx_kernels::Shape out_shape;
  out_shape.assign(x.shape.size(), 1);
  out_shape[0] = N;
  out_shape[1] = C;

  const size_t out_n_bytes = static_cast<size_t>(N * C) * sizeof(float);
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes,
                                rt ? rt->allocator() : nullptr);

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.mutable_bytes());

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t base = (n * C + c) * spatial;
      float val = px[base];
      for (int64_t s = 1; s < spatial; ++s) {
        val = std::max(val, px[base + s]);
      }
      py[n * C + c] = val;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// GlobalLpPool
// ---------------------------------------------------------------------------

Tensor GlobalLpPool::operator()(const Tensor &x, int64_t p, RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::GlobalLpPool: x must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2,
                      "kernel::GlobalLpPool: x must have rank >= 2 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(p >= 1, "kernel::GlobalLpPool: p must be >= 1.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t spatial = SpatialCount(x);

  onnx_kernels::Shape out_shape;
  out_shape.assign(x.shape.size(), 1);
  out_shape[0] = N;
  out_shape[1] = C;

  const size_t out_n_bytes = static_cast<size_t>(N * C) * sizeof(float);
  Tensor out = MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), out_shape, out_n_bytes,
                                rt ? rt->allocator() : nullptr);

  const float *px = x.AsFloat();
  float *py = reinterpret_cast<float *>(out.mutable_bytes());

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t base = (n * C + c) * spatial;
      double acc = 0.0;
      for (int64_t s = 0; s < spatial; ++s) {
        acc += std::pow(std::abs(static_cast<double>(px[base + s])), static_cast<double>(p));
      }
      py[n * C + c] = static_cast<float>(std::pow(acc, 1.0 / static_cast<double>(p)));
    }
  }
  return out;
}

void GlobalAveragePool::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  onnx_kernels::kernel::GlobalAveragePool k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, &rt), rt);
}

void GlobalLpPool::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const int64_t p = GetAttributeIntOrDefault(node, "p", 2);
  onnx_kernels::kernel::GlobalLpPool k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, p, &rt), rt);
}

void GlobalMaxPool::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  onnx_kernels::kernel::GlobalMaxPool k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
