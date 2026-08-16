// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cmath>
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

} // namespace

Tensor LogSoftmax::operator()(const Tensor &x, int64_t axis, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * sizeof(float);
  Tensor y = MakeOutputTensor(DataType::FLOAT, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(x, axis, y);
  return y;
}

void LogSoftmax::operator()(const Tensor &x, int64_t axis, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT,
                      "kernel::LogSoftmax only supports FLOAT tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::FLOAT,
                      "kernel::LogSoftmax preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::LogSoftmax preallocated output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.mutable_bytes() != x.bytes(),
                      "kernel::LogSoftmax does not support aliasing input/output buffers.");

  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(float);
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
  const int64_t axis_dim = x.shape[static_cast<size_t>(resolved_axis)];
  int64_t inner = 1;
  for (int64_t d = resolved_axis + 1; d < rank; ++d) {
    inner *= x.shape[static_cast<size_t>(d)];
  }

  const float *px = x.AsFloat();
  float *py = output.AsFloat();
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      float max_v = -std::numeric_limits<float>::infinity();
      for (int64_t a = 0; a < axis_dim; ++a) {
        const int64_t offset = (o * axis_dim + a) * inner + i;
        max_v = std::max(max_v, px[static_cast<size_t>(offset)]);
      }
      float sum = 0.0f;
      for (int64_t a = 0; a < axis_dim; ++a) {
        const int64_t offset = (o * axis_dim + a) * inner + i;
        sum += std::exp(px[static_cast<size_t>(offset)] - max_v);
      }
      const float log_sum = std::log(sum) + max_v;
      for (int64_t a = 0; a < axis_dim; ++a) {
        const int64_t offset = (o * axis_dim + a) * inner + i;
        py[static_cast<size_t>(offset)] = px[static_cast<size_t>(offset)] - log_sum;
      }
    }
  }
}

void LogSoftmax::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const int64_t axis = GetAttributeIntOrDefault(node, "axis", -1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, axis, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
