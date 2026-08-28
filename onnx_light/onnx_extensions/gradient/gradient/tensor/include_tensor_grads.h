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
 *
 * @code
 * A --> Identity --> C
 *
 * dC --> Identity --> dA
 * @endcode
 */
bool GradIdentity(const NodeProto &node, const std::string &output_grad,
                  std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                  FunctionProto &func);

/**
 * Applies the backward rule for the Reshape operator.
 *
 * C = Reshape(A, shape)  →  dA = Reshape(dC, Shape(A)).
 *
 * @code
 * A, shape --> Reshape ----------> C
 *
 * dC, Shape(A) --> Reshape ------> dA
 * @endcode
 */
bool GradReshape(const NodeProto &node, const std::string &output_grad,
                 std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                 FunctionProto &func);

/**
 * Applies the backward rule for the Transpose operator.
 *
 * C = Transpose(A, perm)  →  dA = Transpose(dC, inverse_perm).
 *
 * @code
 * A, perm --> Transpose ----------> C
 *
 * dC, inverse_perm --> Transpose -> dA
 * @endcode
 */
bool GradTranspose(const NodeProto &node, const std::string &output_grad,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
