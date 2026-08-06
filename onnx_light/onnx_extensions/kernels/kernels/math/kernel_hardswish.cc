// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_core/runtime/parallel_for.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::HardSwish";

// HardSwish parameters are fixed by the ONNX spec: alpha = 1/6, beta = 0.5.
constexpr float kHardSwishAlpha = 1.0f / 6.0f;
constexpr float kHardSwishBeta = 0.5f;

inline float HardSwishOp(float v) {
  const float hs = std::max(0.0f, std::min(1.0f, kHardSwishAlpha * v + kHardSwishBeta));
  return v * hs;
}

} // namespace

Tensor HardSwish::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor y = MakeOutputTensor(x.data_type, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(x, y);
  return y;
}

void HardSwish::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(output.data_type == x.data_type, kName,
                      ": output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape, kName, ": output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == x.size_bytes(), kName,
                      ": output buffer size mismatch.");
  const int64_t n = x.element_count();
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT: {
    const float *px = x.AsFloat();
    float *py = output.AsFloat();
    ParallelFor(n, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[static_cast<size_t>(i)] = HardSwishOp(px[i]);
      }
    });
    return;
  }
  case DataType::FLOAT16:
    detail::UnaryHalfElementwise(x, output, Float16BitsToFloat, FloatToFloat16Bits, HardSwishOp);
    return;
  case DataType::BFLOAT16:
    detail::UnaryHalfElementwise(x, output, Bfloat16BitsToFloat, FloatToBfloat16Bits, HardSwishOp);
    return;
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, and BFLOAT16 tensors.");
  }
}

void HardSwish::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
