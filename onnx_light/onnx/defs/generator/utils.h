// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file generator/utils.h
 * @brief Declares shape-inference helpers for generator-style operators.
 *
 * This header provides the Constant operator inference helper and the
 * compute_output_dim_for_range template used by schema definitions in the
 * generator domain.
 */

#pragma once

#include <algorithm>
#include <cmath>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_util.h"

namespace ONNX_LIGHT_NAMESPACE {

void ConstantOpInference(InferenceContext &ctx);

/**
 * Computes the number of elements produced by a Range operator.
 *
 * Implements the formula ceil((limit - start) / delta), clamped to zero for
 * non-positive results.  All three inputs must be scalar TensorProto values
 * (empty dims).
 *
 * @tparam T Numeric element type (float, double, int32_t, or int64_t).
 * @param start Scalar TensorProto for the range start.
 * @param limit Scalar TensorProto for the exclusive upper bound.
 * @param delta Scalar TensorProto for the step size.
 *
 * Returns:
 *     The number of elements in the output tensor.
 */
template <typename T>
inline int64_t compute_output_dim_for_range(const TensorProto *start, const TensorProto *limit,
                                            const TensorProto *delta) {
  if (!start->dims().empty() || !limit->dims().empty() || !delta->dims().empty()) {
    fail_shape_inference(
        "Input to 'Range' op should be scalars (Tensor with only one element and shape empty)");
  }

  const auto start_data = ParseData<T>(start);
  const auto limit_data = ParseData<T>(limit);
  const auto delta_data = ParseData<T>(delta);

  int64_t n =
      static_cast<int64_t>(ceil((1.0 * (limit_data[0] - start_data[0])) / delta_data[0]));

  n = std::max<int64_t>(n, 0);

  return n;
}

} // namespace ONNX_LIGHT_NAMESPACE
