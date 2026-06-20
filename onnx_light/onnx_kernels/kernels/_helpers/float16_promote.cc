// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/_helpers/float16_promote.h"

#include "onnx_kernels/kernels/_helpers/cast_helper.h"

#include "onnx_light_helpers.h"
#include <cstring>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

Tensor PromoteToFloat32(const Tensor &src) {
  const int32_t dt = src.data_type;
  if (dt == static_cast<int32_t>(DataType::FLOAT) || dt == static_cast<int32_t>(DataType::DOUBLE)) {
    return src;
  }

  const int64_t n = src.element_count();
  const uint8_t *raw = src.bytes();
  std::vector<uint8_t> out(static_cast<size_t>(n) * sizeof(float));
  float *dst = reinterpret_cast<float *>(out.data());

  if (dt == static_cast<int32_t>(DataType::FLOAT16)) {
    const uint16_t *fp16 = reinterpret_cast<const uint16_t *>(raw);
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = kernel::Float16BitsToFloat(fp16[i]);
    }
  } else if (dt == static_cast<int32_t>(DataType::BFLOAT16)) {
    const uint16_t *bf16 = reinterpret_cast<const uint16_t *>(raw);
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = kernel::Bfloat16BitsToFloat(bf16[i]);
    }
  } else {
    throw std::invalid_argument("PromoteToFloat32: unsupported data_type " + std::to_string(dt));
  }

  Tensor result;
  result.name = src.name;
  result.data_type = static_cast<int32_t>(DataType::FLOAT);
  result.shape = src.shape;
  result.data = std::move(out);
  return result;
}

Tensor DemoteFromFloat32(const Tensor &src, int32_t target_dtype) {
  EXT_ENFORCE_INVALID(src.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "DemoteFromFloat32: source must be FLOAT, got ",
                      std::to_string(src.data_type));

  const int64_t n = src.element_count();
  const float *fp32 = reinterpret_cast<const float *>(src.bytes());
  std::vector<uint8_t> out(static_cast<size_t>(n) * sizeof(uint16_t));
  uint16_t *dst = reinterpret_cast<uint16_t *>(out.data());

  if (target_dtype == static_cast<int32_t>(DataType::FLOAT16)) {
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = kernel::FloatToFloat16Bits(fp32[i]);
    }
  } else if (target_dtype == static_cast<int32_t>(DataType::BFLOAT16)) {
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = kernel::FloatToBfloat16Bits(fp32[i]);
    }
  } else {
    throw std::invalid_argument("DemoteFromFloat32: unsupported target_dtype " +
                                std::to_string(target_dtype));
  }

  Tensor result;
  result.name = src.name;
  result.data_type = target_dtype;
  result.shape = src.shape;
  result.data = std::move(out);
  return result;
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
