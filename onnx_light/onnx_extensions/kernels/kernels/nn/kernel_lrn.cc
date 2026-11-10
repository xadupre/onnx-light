// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor LRN::operator()(const Tensor &x, int64_t size, float alpha, float beta, float bias,
                       RuntimeContext *rt) const {
  const bool half = IsHalfPrecision(x.data_type);
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT) || half,
                      "kernel::LRN: x must be FLOAT, FLOAT16 or BFLOAT16.");
  EXT_ENFORCE_INVALID(x.shape.size() >= 2, "kernel::LRN: x must have rank >= 2 (N, C, D1, ...).");
  EXT_ENFORCE_INVALID(size > 0, "kernel::LRN: size must be strictly positive.");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  // Spatial element count per (n, c) slice.
  int64_t spatial = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) {
    spatial *= x.shape[i];
  }
  const size_t out_n_bytes = x.size_bytes();
  Tensor out = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x.data_type, x.shape, out_n_bytes, nullptr);

  const auto decode = x.data_type == static_cast<int32_t>(DataType::BFLOAT16) ? Bfloat16BitsToFloat
                                                                              : Float16BitsToFloat;
  const auto encode = x.data_type == static_cast<int32_t>(DataType::BFLOAT16) ? FloatToBfloat16Bits
                                                                              : FloatToFloat16Bits;
  const float *px = half ? nullptr : x.AsFloat();
  const uint16_t *px_half = half ? reinterpret_cast<const uint16_t *>(x.bytes()) : nullptr;
  const auto read = [&](int64_t i) { return half ? decode(px_half[i]) : px[i]; };
  float *py = half ? nullptr : out.AsFloat();
  uint16_t *py_half = half ? reinterpret_cast<uint16_t *>(out.mutable_bytes()) : nullptr;

  // For each (n, c, s) compute sum of squares over the channel window then
  // produce the normalized output. The window is centered on c with extent
  // [max(0, c - floor((size-1)/2)), min(C-1, c + ceil((size-1)/2))].
  const int64_t half_lo = (size - 1) / 2; // floor((size - 1) / 2)
  const int64_t half_hi = size / 2;       // ceil((size - 1) / 2)
  const double scale = static_cast<double>(alpha) / static_cast<double>(size);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      const int64_t c_begin = std::max<int64_t>(0, c - half_lo);
      const int64_t c_end = std::min<int64_t>(C - 1, c + half_hi);
      for (int64_t s = 0; s < spatial; ++s) {
        double sq_sum = 0.0;
        for (int64_t i = c_begin; i <= c_end; ++i) {
          const double v = static_cast<double>(read((n * C + i) * spatial + s));
          sq_sum += v * v;
        }
        const double denom =
            std::pow(static_cast<double>(bias) + scale * sq_sum, static_cast<double>(beta));
        const int64_t index = (n * C + c) * spatial + s;
        const float y = static_cast<float>(static_cast<double>(read(index)) / denom);
        if (half) {
          py_half[index] = encode(y);
        } else {
          py[index] = y;
        }
      }
    }
  }
  return out;
}

void LRN::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt);
  const int64_t size = GetRequiredAttributeInt(node, "size");
  const float alpha = GetAttributeFloatOrDefault(node, "alpha", 0.0001f);
  const float beta = GetAttributeFloatOrDefault(node, "beta", 0.75f);
  const float bias = GetAttributeFloatOrDefault(node, "bias", 1.0f);
  onnx_kernels::kernel::LRN k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, size, alpha, beta, bias, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
