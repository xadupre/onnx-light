// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

int64_t ResolveAxis(int64_t axis, int64_t rank) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      "kernel::LogSoftmax: axis is out of range.");
  return resolved;
}

template <typename T>
void LogSoftmaxTyped(const Tensor &x, Tensor &output, int64_t outer, int64_t axis_dim,
                     int64_t inner) {
  const T *px = x.As<T>();
  T *py = output.As<T>();
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      T max_v = -std::numeric_limits<T>::infinity();
      for (int64_t a = 0; a < axis_dim; ++a) {
        const int64_t offset = (o * axis_dim + a) * inner + i;
        max_v = std::max(max_v, px[offset]);
      }
      T sum = 0;
      for (int64_t a = 0; a < axis_dim; ++a) {
        const int64_t offset = (o * axis_dim + a) * inner + i;
        sum += std::exp(px[offset] - max_v);
      }
      const T log_sum = std::log(sum);
      for (int64_t a = 0; a < axis_dim; ++a) {
        const int64_t offset = (o * axis_dim + a) * inner + i;
        py[offset] = (px[offset] - max_v) - log_sum;
      }
    }
  }
}

} // namespace

Tensor LogSoftmax::operator()(const Tensor &x, int64_t axis, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * ElementSize(x.data_type);
  Tensor y = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, y_n_bytes)
                : MakeOutputTensor(x.data_type, x.shape, y_n_bytes, nullptr);
  (*this)(x, axis, y);
  return y;
}

void LogSoftmax::operator()(const Tensor &x, int64_t axis, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT || x.data_type == DataType::DOUBLE ||
                          x.data_type == DataType::FLOAT16,
                      "kernel::LogSoftmax only supports FLOAT, DOUBLE and FLOAT16 tensors.");
  EXT_ENFORCE_INVALID(output.data_type == x.data_type,
                      "kernel::LogSoftmax preallocated output dtype must match input.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::LogSoftmax preallocated output shape must match input shape.");
  EXT_ENFORCE_INVALID(x.element_count() == 0 || output.mutable_bytes() != x.bytes(),
                      "kernel::LogSoftmax does not support aliasing input/output buffers.");

  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n) * ElementSize(x.data_type);
  EXT_ENFORCE_INVALID(
      output.size_bytes() == expected_bytes,
      "kernel::LogSoftmax preallocated output buffer has unexpected size in bytes.");

  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, "kernel::LogSoftmax: input rank must be >= 1.");
  const int64_t resolved_axis = ResolveAxis(axis, rank);

  int64_t outer = 1;
  for (int64_t d = 0; d < resolved_axis; ++d) {
    outer *= x.shape[static_cast<size_t>(d)];
  }
  int64_t axis_dim = x.shape[static_cast<size_t>(resolved_axis)];
  int64_t inner = 1;
  for (int64_t d = resolved_axis + 1; d < rank; ++d) {
    inner *= x.shape[static_cast<size_t>(d)];
  }

  if (ctx_.opset.version > 0 && ctx_.opset.version < 13) {
    axis_dim *= inner;
    inner = 1;
  }
  if (n == 0) {
    return;
  }
  if (x.data_type == DataType::FLOAT16) {
    Tensor promoted = core::runtime::PromoteToFloat32(x);
    Tensor result = MakeOutputTensor(DataType::FLOAT, x.shape, n * sizeof(float), nullptr);
    LogSoftmaxTyped<float>(promoted, result, outer, axis_dim, inner);
    Tensor demoted = core::runtime::DemoteFromFloat32(result, x.data_type);
    std::memcpy(output.mutable_bytes(), demoted.bytes(), output.size_bytes());
  } else if (x.data_type == DataType::DOUBLE) {
    LogSoftmaxTyped<double>(x, output, outer, axis_dim, inner);
  } else {
    LogSoftmaxTyped<float>(x, output, outer, axis_dim, inner);
  }
}

void LogSoftmax::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const int64_t axis = GetAttributeIntOrDefault(
      node, "axis", ctx_.opset.version > 0 && ctx_.opset.version < 13 ? 1 : -1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, axis, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
