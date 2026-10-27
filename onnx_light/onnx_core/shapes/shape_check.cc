// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_check.h"
#include "onnx_lib/common/safe_math.h"

namespace ONNX_LIGHT_NAMESPACE::core::shapes {

int64_t CheckedDimensionMultiply(int64_t lhs, int64_t rhs) {
  int64_t result = 0;
  EXT_ENFORCE_INVALID(!checked_mul_overflow(lhs, rhs, &result),
                      "Integer overflow while multiplying dimension values ", lhs, " and ", rhs,
                      ".");
  return result;
}

int64_t CheckedDimensionAdd(int64_t lhs, int64_t rhs) {
  int64_t result = 0;
  EXT_ENFORCE_INVALID(!checked_add_overflow(lhs, rhs, &result),
                      "Integer overflow while adding dimension values ", lhs, " and ", rhs, ".");
  return result;
}

void CheckNodeOpAndOutput(const NodeProto &node, const char *expected_op_type, const char *caller) {
  EXT_ENFORCE_INVALID(node.op_type() == expected_op_type, caller, " expects op_type='",
                      expected_op_type, "', got '", node.op_type(), "'.");
  EXT_ENFORCE_INVALID(node.output_size() >= 1, caller, ": node has no output.");
}

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes
