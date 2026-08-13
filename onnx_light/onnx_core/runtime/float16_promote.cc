// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/float16_promote.h"

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"

#include "onnx_light_helpers.h"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

Tensor PromoteToFloat32(const Tensor &src, RuntimeContext *rt) {
  return PromoteToFloat32(src, rt, std::numeric_limits<int64_t>::max());
}

Tensor PromoteToFloat32(const Tensor &src, RuntimeContext *rt, int64_t parallel_minimum_elements) {
  const int32_t dt = src.data_type;
  if (dt == static_cast<int32_t>(DataType::FLOAT) || dt == static_cast<int32_t>(DataType::DOUBLE)) {
    return src;
  }

  const int64_t n = src.element_count();
  const uint8_t *raw = src.bytes();
  Tensor result =
      MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), src.shape,
                       static_cast<size_t>(n) * sizeof(float), rt ? rt->allocator() : nullptr);
  result.name = src.name;
  float *dst = reinterpret_cast<float *>(result.mutable_bytes());

  if (dt == static_cast<int32_t>(DataType::FLOAT16)) {
    const uint16_t *fp16 = reinterpret_cast<const uint16_t *>(raw);
    ParallelFor(n, parallel_minimum_elements, [fp16, dst](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        dst[i] = Float16BitsToFloat(fp16[i]);
      }
    });
  } else if (dt == static_cast<int32_t>(DataType::BFLOAT16)) {
    const uint16_t *bf16 = reinterpret_cast<const uint16_t *>(raw);
    ParallelFor(n, parallel_minimum_elements, [bf16, dst](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        dst[i] = Bfloat16BitsToFloat(bf16[i]);
      }
    });
  } else {
    EXT_THROW_INVALID("PromoteToFloat32: unsupported data_type ", dt);
  }

  return result;
}

Tensor DemoteFromFloat32(const Tensor &src, int32_t target_dtype, RuntimeContext *rt) {
  return DemoteFromFloat32(src, target_dtype, rt, std::numeric_limits<int64_t>::max());
}

Tensor DemoteFromFloat32(const Tensor &src, int32_t target_dtype, RuntimeContext *rt,
                         int64_t parallel_minimum_elements) {
  EXT_ENFORCE_INVALID(src.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "DemoteFromFloat32: source must be FLOAT, got ", src.data_type);

  const int64_t n = src.element_count();
  const float *fp32 = reinterpret_cast<const float *>(src.bytes());
  Tensor result =
      MakeOutputTensor(target_dtype, src.shape, static_cast<size_t>(n) * sizeof(uint16_t),
                       rt ? rt->allocator() : nullptr);
  result.name = src.name;
  uint16_t *dst = reinterpret_cast<uint16_t *>(result.mutable_bytes());

  if (target_dtype == static_cast<int32_t>(DataType::FLOAT16)) {
    ParallelFor(n, parallel_minimum_elements, [fp32, dst](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        dst[i] = FloatToFloat16Bits(fp32[i]);
      }
    });
  } else if (target_dtype == static_cast<int32_t>(DataType::BFLOAT16)) {
    ParallelFor(n, parallel_minimum_elements, [fp32, dst](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        dst[i] = FloatToBfloat16Bits(fp32[i]);
      }
    });
  } else {
    EXT_THROW_INVALID("DemoteFromFloat32: unsupported target_dtype ", target_dtype);
  }

  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
