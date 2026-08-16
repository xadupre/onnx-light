// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Utility helpers for promoting FLOAT16/BFLOAT16 tensors to FLOAT32 and
// demoting back. Intended for use inside individual kernels that need to
// support half-precision inputs by computing in float32 internally.

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

class RuntimeContext;

/// Returns true if the data type is FLOAT16 or BFLOAT16.
inline bool IsHalfPrecision(int32_t data_type) {
  return data_type == static_cast<int32_t>(DataType::FLOAT16) ||
         data_type == static_cast<int32_t>(DataType::BFLOAT16);
}

/// Converts a FLOAT16 or BFLOAT16 tensor to FLOAT32.
/// If the input is already FLOAT or DOUBLE, returns a copy unchanged (no conversion).
Tensor PromoteToFloat32(const Tensor &src, RuntimeContext *rt = nullptr);

/// Converts a half-precision tensor to FLOAT32 with a kernel-specific parallel threshold.
Tensor PromoteToFloat32(const Tensor &src, RuntimeContext *rt, int64_t parallel_minimum_elements);

/// Demotes a FLOAT32 tensor back to FLOAT16 or BFLOAT16.
/// @param src The float32 tensor to demote.
/// @param target_dtype The target data type (FLOAT16 or BFLOAT16).
/// @return A new Tensor with data_type == target_dtype and the same shape/name.
Tensor DemoteFromFloat32(const Tensor &src, int32_t target_dtype, RuntimeContext *rt = nullptr);

/// Demotes FLOAT32 with a kernel-specific parallel threshold.
Tensor DemoteFromFloat32(const Tensor &src, int32_t target_dtype, RuntimeContext *rt,
                         int64_t parallel_minimum_elements);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
