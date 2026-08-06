// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/parallel_for.h"

#include "onnx_core/runtime/cast_helper.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Relu";

template <typename T> void ComputeInPlace(const Tensor &x, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  ParallelFor(n, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const T v = px[i];
      py[i] = v > static_cast<T>(0) ? v : static_cast<T>(0);
    }
  });
}

void ComputeFloat16(const Tensor &x, Tensor &output) {
  const int64_t n = x.element_count();
  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
  ParallelFor(n, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const float v = Float16BitsToFloat(px[i]);
      py[i] = FloatToFloat16Bits(v > 0.0f ? v : 0.0f);
    }
  });
}

void ComputeBfloat16(const Tensor &x, Tensor &output) {
  const int64_t n = x.element_count();
  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
  ParallelFor(n, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const float v = Bfloat16BitsToFloat(px[i]);
      py[i] = FloatToBfloat16Bits(v > 0.0f ? v : 0.0f);
    }
  });
}

void Dispatch(const Tensor &x, Tensor &output) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    ComputeInPlace<float>(x, output);
    return;
  case DataType::DOUBLE:
    ComputeInPlace<double>(x, output);
    return;
  case DataType::INT8:
    ComputeInPlace<int8_t>(x, output);
    return;
  case DataType::INT16:
    ComputeInPlace<int16_t>(x, output);
    return;
  case DataType::INT32:
    ComputeInPlace<int32_t>(x, output);
    return;
  case DataType::INT64:
    ComputeInPlace<int64_t>(x, output);
    return;
  case DataType::FLOAT16:
    ComputeFloat16(x, output);
    return;
  case DataType::BFLOAT16:
    ComputeBfloat16(x, output);
    return;
  default:
    EXT_THROW_INVALID(
        kName, ": unsupported data type ", x.data_type,
        ", only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, and signed integer tensors.");
  }
}

void ValidateOutput(const Tensor &x, const Tensor &output) {
  EXT_ENFORCE_INVALID(output.data_type == x.data_type, kName,
                      ": output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape, kName, ": output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == x.size_bytes(), kName,
                      ": output buffer size mismatch.");
}

} // namespace

Tensor Relu::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor out = MakeOutputTensor(x.data_type, x.shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  Dispatch(x, out);
  return out;
}

void Relu::operator()(const Tensor &x, Tensor &output) const {
  ValidateOutput(x, output);
  Dispatch(x, output);
}

void Relu::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
