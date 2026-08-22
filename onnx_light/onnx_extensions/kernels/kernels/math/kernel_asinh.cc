// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cmath>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Asinh";
constexpr uint32_t kTuningAbi = 2;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;

constexpr std::array<int32_t, 3> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::FLOAT16),
    static_cast<int32_t>(DataType::BFLOAT16)};

} // namespace

Asinh::Asinh(const KernelContext &ctx) : KernelBase(ctx), tuning_(kPortableParallelMinimum) {}

void Asinh::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Asinh", kSupportedElementTypes, kPortableParallelMinimum,
                                        kTuningAbi);
}

KernelTuningKey Asinh::TuningKey(int32_t element_type) const {
  return tuning::IsSupportedElementType(element_type, kSupportedElementTypes)
             ? tuning::MakePortableTuningKey("Asinh", element_type, kTuningAbi)
             : KernelTuningKey{};
}

void Asinh::Configure(const KernelTuningParameters &parameters) {
  tuning::ConfigureParallelTuning("Asinh", parameters, tuning_, kTuningAbi);
}

Tensor Asinh::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor y = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, y_n_bytes)
                : MakeOutputTensor(x.data_type, x.shape, y_n_bytes, nullptr);
  (*this)(x, y);
  return y;
}

void Asinh::operator()(const Tensor &x, Tensor &output) const {
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
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[static_cast<size_t>(i)] = std::asinh(px[i]);
      }
    });
    return;
  }
  case DataType::FLOAT16:
    detail::UnaryHalfElementwise(x, output, Float16BitsToFloat, FloatToFloat16Bits,
                                 tuning_.parallel_minimum_elements,
                                 [](float v) { return std::asinh(v); });
    return;
  case DataType::BFLOAT16:
    detail::UnaryHalfElementwise(x, output, Bfloat16BitsToFloat, FloatToBfloat16Bits,
                                 tuning_.parallel_minimum_elements,
                                 [](float v) { return std::asinh(v); });
    return;
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, and BFLOAT16 tensors.");
  }
}

void Asinh::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
