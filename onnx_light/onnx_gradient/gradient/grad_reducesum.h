// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"
#include <string>
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

/**
 * Applies the backward rule for the ReduceSum operator.
 *
 * Y = ReduceSum(X)  →  dX = Expand(dY, Shape(X))
 */
bool GradReduceSum(const NodeProto &node, const std::string &output_grad,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func);

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
