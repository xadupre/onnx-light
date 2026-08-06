// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"
#include <string>
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

/**
 * Applies the backward rule for the Identity operator.
 *
 * C = A  →  dA = dC.
 */
bool GradIdentity(const NodeProto &node, const std::string &output_grad,
                  std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                  FunctionProto &func);

/**
 * Applies the backward rule for the Reshape operator.
 *
 * C = Reshape(A, shape)  →  dA = Reshape(dC, Shape(A)).
 */
bool GradReshape(const NodeProto &node, const std::string &output_grad,
                 std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                 FunctionProto &func);

/**
 * Applies the backward rule for the Transpose operator.
 *
 * C = Transpose(A, perm)  →  dA = Transpose(dC, inverse_perm).
 */
bool GradTranspose(const NodeProto &node, const std::string &output_grad,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
