// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/parallel_for.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Abs";
constexpr int64_t kAbsParallelGrainSize = 32 * kParallelForGrainSize;

} // namespace

Tensor Abs::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor y = MakeOutputTensor(x.data_type, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(x, y);
  return y;
}

void Abs::operator()(const Tensor &x, Tensor &output) const {
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
    ParallelFor(n, kAbsParallelGrainSize, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = std::fabs(px[i]);
      }
    });
    return;
  }
  case DataType::DOUBLE: {
    const double *px = x.AsDouble();
    double *py = output.AsDouble();
    ParallelFor(n, kAbsParallelGrainSize, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = std::fabs(px[i]);
      }
    });
    return;
  }
  case DataType::FLOAT16: {
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
    ParallelFor(n, kAbsParallelGrainSize, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = FloatToFloat16Bits(std::fabs(Float16BitsToFloat(px[i])));
      }
    });
    return;
  }
  case DataType::BFLOAT16: {
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
    ParallelFor(n, kAbsParallelGrainSize, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = FloatToBfloat16Bits(std::fabs(Bfloat16BitsToFloat(px[i])));
      }
    });
    return;
  }
  case DataType::INT8: {
    const int8_t *px = x.AsInt8();
    int8_t *py = output.AsInt8();
    ParallelFor(n, kAbsParallelGrainSize, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const int32_t v = static_cast<int32_t>(px[i]);
        py[i] = static_cast<int8_t>(v < 0 ? -v : v);
      }
    });
    return;
  }
  case DataType::INT16: {
    const int16_t *px = x.AsInt16();
    int16_t *py = output.AsInt16();
    ParallelFor(n, kAbsParallelGrainSize, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const int32_t v = static_cast<int32_t>(px[i]);
        py[i] = static_cast<int16_t>(v < 0 ? -v : v);
      }
    });
    return;
  }
  case DataType::INT32: {
    const int32_t *px = x.AsInt32();
    int32_t *py = output.AsInt32();
    ParallelFor(n, kAbsParallelGrainSize, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const int64_t v = static_cast<int64_t>(px[i]);
        py[i] = static_cast<int32_t>(v < 0 ? -v : v);
      }
    });
    return;
  }
  case DataType::INT64: {
    const int64_t *px = x.AsInt64();
    int64_t *py = output.AsInt64();
    ParallelFor(n, kAbsParallelGrainSize, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const uint64_t u = static_cast<uint64_t>(px[i]);
        py[i] = static_cast<int64_t>(px[i] < 0 ? (~u + 1) : u);
      }
    });
    return;
  }
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32, and "
                      "INT64 tensors.");
  }
}

void Abs::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
