// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_check.h"

#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {

void CheckNodeOpAndOutput(const NodeProto &node, const char *expected_op_type, const char *caller) {
  if (node.op_type() != expected_op_type) {
    throw std::invalid_argument(std::string(caller) + " expects op_type='" + expected_op_type +
                                "', got '" + node.op_type().as_string() + "'.");
  }
  if (node.output_size() < 1) {
    throw std::invalid_argument(std::string(caller) + ": node has no output.");
  }
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
