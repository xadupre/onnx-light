// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Round";
constexpr uint32_t kTuningAbi = 2;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;

constexpr std::array<int32_t, 3> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT), static_cast<int32_t>(DataType::FLOAT16),
    static_cast<int32_t>(DataType::BFLOAT16)};

using DecodeFunc = float (*)(uint16_t);
using EncodeFunc = uint16_t (*)(float);

// ONNX Round rounds halves to the nearest even integer (banker's rounding).
// std::nearbyint honors the current rounding mode, so each parallel block pins
// it to FE_TONEAREST (round-to-nearest, ties-to-even) per IEEE 754 and restores
// the previous mode afterwards.
void RoundHalf(const Tensor &x, Tensor &output, DecodeFunc decode, EncodeFunc encode,
               int64_t grain_size) {
  const int64_t n = x.element_count();
  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
  ParallelFor(n, grain_size, [px, py, decode, encode](int64_t begin, int64_t end) {
    const int previous_rounding_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    for (int64_t i = begin; i < end; ++i) {
      py[i] = encode(std::nearbyint(decode(px[i])));
    }
    std::fesetround(previous_rounding_mode);
  });
}

} // namespace

Round::Round(const KernelContext &ctx)
    : ParallelTunableKernel(ctx, "Round", kSupportedElementTypes, kPortableParallelMinimum,
                            kTuningAbi) {}

void Round::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Round", kSupportedElementTypes, kPortableParallelMinimum,
                                        kTuningAbi);
}

Tensor Round::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor y = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, y_n_bytes)
                : MakeOutputTensor(x.data_type, x.shape, y_n_bytes, nullptr);
  (*this)(x, y);
  return y;
}

void Round::operator()(const Tensor &x, Tensor &output) const {
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
    ParallelFor(n, tuning().parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      const int previous_rounding_mode = std::fegetround();
      std::fesetround(FE_TONEAREST);
      for (int64_t i = begin; i < end; ++i) {
        py[static_cast<size_t>(i)] = std::nearbyint(px[i]);
      }
      std::fesetround(previous_rounding_mode);
    });
    return;
  }
  case DataType::FLOAT16:
    RoundHalf(x, output, Float16BitsToFloat, FloatToFloat16Bits,
              tuning().parallel_minimum_elements);
    return;
  case DataType::BFLOAT16:
    RoundHalf(x, output, Bfloat16BitsToFloat, FloatToBfloat16Bits,
              tuning().parallel_minimum_elements);
    return;
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, FLOAT16, and BFLOAT16 tensors.");
  }
}

void Round::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
