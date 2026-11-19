// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <array>
#include <cmath>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {
namespace {
constexpr const char *kName = "kernel::GeGLU";
constexpr std::array<int32_t, 4> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16)};

template <typename T> T GeGLUOp(T a, T b, bool approximate) {
  const T half = static_cast<T>(0.5);
  const T one = static_cast<T>(1);
  if (approximate) {
    const T inner =
        static_cast<T>(0.7978845608028654L) * (a + static_cast<T>(0.044715) * a * a * a);
    return (half * a * (one + std::tanh(inner))) * b;
  }
  const T inv_sqrt2 = static_cast<T>(1.0L / 1.4142135623730951L);
  return (half * a * (one + std::erf(a * inv_sqrt2))) * b;
}

void ValidateInputs(const Tensor &a, const Tensor &b, const std::string &approximate) {
  EXT_ENFORCE_INVALID(approximate == "none" || approximate == "tanh", kName,
                      ": 'approximate' must be 'none' or 'tanh'.");
  EXT_ENFORCE_INVALID(a.shape == b.shape, kName,
                      ": inputs A and B must have identical shapes (no broadcasting).");
  EXT_ENFORCE_INVALID(a.data_type == DataType::FLOAT || a.data_type == DataType::DOUBLE ||
                          a.data_type == DataType::FLOAT16 || a.data_type == DataType::BFLOAT16,
                      kName, ": only supports FLOAT, DOUBLE, FLOAT16 and BFLOAT16 tensors.");
  EXT_ENFORCE_INVALID(a.data_type == b.data_type, kName, ": input dtypes must match.");
  const int64_t count = a.shape.product(0, a.shape.size(), kName);
  const size_t bytes = PackedByteSize(a.data_type, count);
  EXT_ENFORCE_INVALID(a.size_bytes() == bytes && b.size_bytes() == bytes, kName,
                      ": input buffer size mismatch.");
}

template <typename T, typename Op>
void Compute(const Tensor &a, const Tensor &b, Tensor &output, int64_t grain, Op op) {
  const T *pa = reinterpret_cast<const T *>(a.bytes());
  const T *pb = reinterpret_cast<const T *>(b.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  ParallelFor(a.element_count(), grain, [pa, pb, py, op](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      py[i] = op(pa[i], pb[i]);
    }
  });
}
} // namespace

GeGLU::GeGLU(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "GeGLU", kSupportedElementTypes, kParallelForGrainSize) {}

void GeGLU::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("GeGLU", kSupportedElementTypes, kParallelForGrainSize);
}

Tensor GeGLU::operator()(const Tensor &a, const Tensor &b, const std::string &approximate,
                         RuntimeContext *rt) const {
  ValidateInputs(a, b, approximate);
  const size_t bytes = a.size_bytes();
  Tensor output = rt ? rt->MakeOutputTensor(0, a.data_type, a.shape, bytes)
                     : MakeOutputTensor(a.data_type, a.shape, bytes, nullptr);
  (*this)(a, b, approximate, output);
  return output;
}

void GeGLU::operator()(const Tensor &a, const Tensor &b, const std::string &approximate,
                       Tensor &output) const {
  ValidateInputs(a, b, approximate);
  detail::CheckPreallocatedOutput(kName, "matching", a.data_type, a.shape, a.size_bytes(), output);
  const bool tanh_approx = approximate == "tanh";
  const int64_t grain = tuning().parallel_minimum_elements;
  switch (a.data_type) {
  case DataType::FLOAT:
    return Compute<float>(a, b, output, grain,
                          [tanh_approx](float x, float y) { return GeGLUOp(x, y, tanh_approx); });
  case DataType::DOUBLE:
    return Compute<double>(a, b, output, grain, [tanh_approx](double x, double y) {
      return GeGLUOp(x, y, tanh_approx);
    });
  case DataType::FLOAT16:
    return Compute<uint16_t>(a, b, output, grain, [tanh_approx](uint16_t x, uint16_t y) {
      return FloatToFloat16Bits(GeGLUOp(Float16BitsToFloat(x), Float16BitsToFloat(y), tanh_approx));
    });
  case DataType::BFLOAT16:
    return Compute<uint16_t>(a, b, output, grain, [tanh_approx](uint16_t x, uint16_t y) {
      return FloatToBfloat16Bits(
          GeGLUOp(Bfloat16BitsToFloat(x), Bfloat16BitsToFloat(y), tanh_approx));
    });
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", a.data_type);
  }
}

void GeGLU::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  const Tensor &a = GetInput(node, 0, rt);
  const Tensor &b = GetInput(node, 1, rt);
  const std::string approximate = GetAttributeStringOrDefault(node, "approximate", "none");
  SetOutput(node, 0, (*this)(a, b, approximate, &rt), rt);
}
} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
