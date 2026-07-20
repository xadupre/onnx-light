// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/dispatch_table.h"

#include <string>
#include <unordered_map>

#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace shapes {

namespace {

// Verifies the node declares at least `expected` inputs.
void RequireInputs(const NodeProto &node, int expected) {
  EXT_ENFORCE_INVALID(node.input_size() >= expected, "ComputeShapeNode: op '", node.op_type(),
                      "' expects at least ", std::to_string(expected), " input(s), got ",
                      std::to_string(node.input_size()), ".");
}

} // namespace

const std::unordered_map<std::string, ComputeShapeFn> &DispatchTable() {
  return std::unordered_map<std::string, ComputeShapeFn>();
}

} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
