// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::core::shapes {

void CheckNodeOpAndOutput(const NodeProto &node, const char *expected_op_type, const char *caller) {
  EXT_ENFORCE_INVALID(node.op_type() == expected_op_type, caller, " expects op_type='",
                      expected_op_type, "', got '", node.op_type(), "'.");
  EXT_ENFORCE_INVALID(node.output_size() >= 1, caller, ": node has no output.");
}

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes
