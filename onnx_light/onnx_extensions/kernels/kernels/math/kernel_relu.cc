// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/parallel_for.h"

#include "onnx_core/runtime/kernels/cast_helper.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Relu";
constexpr uint32_t kTuningAbi = 2;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;

constexpr std::array<int32_t, 8> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT),   static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::INT8),    static_cast<int32_t>(DataType::INT16),
    static_cast<int32_t>(DataType::INT32),   static_cast<int32_t>(DataType::INT64),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16),
};

template <typename T> void ComputeInPlace(const Tensor &x, Tensor &output, int64_t grain_size) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  ParallelFor(n, grain_size, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const T v = px[i];
      py[i] = v > static_cast<T>(0) ? v : static_cast<T>(0);
    }
  });
}

void ComputeFloat16(const Tensor &x, Tensor &output, int64_t grain_size) {
  const int64_t n = x.element_count();
  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
  ParallelFor(n, grain_size, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const float v = Float16BitsToFloat(px[i]);
      py[i] = FloatToFloat16Bits(v > 0.0f ? v : 0.0f);
    }
  });
}

void ComputeBfloat16(const Tensor &x, Tensor &output, int64_t grain_size) {
  const int64_t n = x.element_count();
  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
  ParallelFor(n, grain_size, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const float v = Bfloat16BitsToFloat(px[i]);
      py[i] = FloatToBfloat16Bits(v > 0.0f ? v : 0.0f);
    }
  });
}

void Dispatch(const Tensor &x, Tensor &output, int64_t grain_size) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    ComputeInPlace<float>(x, output, grain_size);
    return;
  case DataType::DOUBLE:
    ComputeInPlace<double>(x, output, grain_size);
    return;
  case DataType::INT8:
    ComputeInPlace<int8_t>(x, output, grain_size);
    return;
  case DataType::INT16:
    ComputeInPlace<int16_t>(x, output, grain_size);
    return;
  case DataType::INT32:
    ComputeInPlace<int32_t>(x, output, grain_size);
    return;
  case DataType::INT64:
    ComputeInPlace<int64_t>(x, output, grain_size);
    return;
  case DataType::FLOAT16:
    ComputeFloat16(x, output, grain_size);
    return;
  case DataType::BFLOAT16:
    ComputeBfloat16(x, output, grain_size);
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

Relu::Relu(const KernelContext &ctx) : KernelBase(ctx), tuning_(kPortableParallelMinimum) {}

void Relu::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Relu", kSupportedElementTypes, kPortableParallelMinimum,
                                        kTuningAbi);
}

KernelTuningKey Relu::TuningKey(int32_t element_type) const {
  return tuning::IsSupportedElementType(element_type, kSupportedElementTypes)
             ? tuning::MakePortableTuningKey("Relu", element_type, kTuningAbi)
             : KernelTuningKey{};
}

void Relu::Configure(const KernelTuningParameters &parameters) {
  tuning::ConfigureParallelTuning("Relu", parameters, tuning_, kTuningAbi);
}

Tensor Relu::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor out = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x.data_type, x.shape, out_n_bytes, nullptr);
  Dispatch(x, out, tuning_.parallel_minimum_elements);
  return out;
}

void Relu::operator()(const Tensor &x, Tensor &output) const {
  ValidateOutput(x, output);
  Dispatch(x, output, tuning_.parallel_minimum_elements);
}

void Relu::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
