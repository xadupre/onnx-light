// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/elementwise_helpers.h"
#include "onnx_core/runtime/parallel_for.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Neg";

// Negates a signed integer tensor using unsigned two's complement arithmetic so
// that the minimum-value edge case (e.g. -INT8_MIN) wraps correctly without UB.
template <typename T> void NegInt(const Tensor &x, Tensor &output) {
  using U = std::make_unsigned_t<T>;
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  ParallelFor(n, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      py[i] = static_cast<T>(U{0} - static_cast<U>(px[i]));
    }
  });
}

} // namespace

Tensor Neg::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor y = MakeOutputTensor(x.data_type, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(x, y);
  return y;
}

void Neg::operator()(const Tensor &x, Tensor &output) const {
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
        py[i] = -px[i];
      }
    });
    return;
  }
  case DataType::DOUBLE: {
    const double *px = x.AsDouble();
    double *py = output.AsDouble();
    ParallelFor(n, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = -px[i];
      }
    });
    return;
  }
  case DataType::FLOAT16:
    detail::UnaryHalfElementwise(x, output, Float16BitsToFloat, FloatToFloat16Bits,
                                 [](float v) { return -v; });
    return;
  case DataType::BFLOAT16:
    detail::UnaryHalfElementwise(x, output, Bfloat16BitsToFloat, FloatToBfloat16Bits,
                                 [](float v) { return -v; });
    return;
  case DataType::INT8:
    NegInt<int8_t>(x, output);
    return;
  case DataType::INT16:
    NegInt<int16_t>(x, output);
    return;
  case DataType::INT32:
    NegInt<int32_t>(x, output);
    return;
  case DataType::INT64:
    NegInt<int64_t>(x, output);
    return;
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32, and "
                      "INT64 tensors.");
  }
}

void Neg::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
