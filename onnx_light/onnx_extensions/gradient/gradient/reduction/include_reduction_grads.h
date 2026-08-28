// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"
#include <string>
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

/**
 * Applies the backward rule for the ReduceMean operator.
 *
 * Y = ReduceMean(X)  →  dX = Expand(dY / Size(X), Shape(X)).
 *
 * @code
 * X, axes --> [ReduceMean] ------------------> Y
 *
 * dY, Size(X) --> [Div] --> [Expand(Shape(X))] --> dX
 * @endcode
 */
bool GradReduceMean(const NodeProto &node, const std::string &output_grad,
                    std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                    FunctionProto &func);

/**
 * Applies the backward rule for the ReduceSum operator.
 *
 * Y = ReduceSum(X)  →  dX = Expand(dY, Shape(X)).
 *
 * @code
 * X, axes --> [ReduceSum] -----------------> Y
 *
 * dY --> [Expand(Shape(X))] -----------------> dX
 * @endcode
 */
bool GradReduceSum(const NodeProto &node, const std::string &output_grad,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
