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
 * Applies the backward rule for the Relu operator.
 *
 * C = relu(A)  →  dA = dC * relu(sign(A)).
 */
bool GradRelu(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func);

/**
 * Applies the backward rule for the Sigmoid operator.
 *
 * C = sigmoid(A)  →  dA = dC * C * (1 - C).
 */
bool GradSigmoid(const NodeProto &node, const std::string &output_grad,
                 std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                 FunctionProto &func);

/**
 * Applies the backward rule for the Tanh operator.
 *
 * C = tanh(A)  →  dA = dC * (1 - C^2).
 */
bool GradTanh(const NodeProto &node, const std::string &output_grad,
              std::unordered_map<std::string, std::string> &grad_accum, int &counter,
              FunctionProto &func);

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
