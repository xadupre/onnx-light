// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"
#include <functional>
#include <string>
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

/**
 * Defines the signature for a per-operator backward (gradient) function.
 *
 * Parameters mirror those of ApplyBackward: @p node is the forward op, @p output_grad
 * is the name of the gradient tensor flowing into this op's output, @p grad_accum
 * accumulates partial input gradients, @p counter generates unique names, and @p func
 * receives the new backward nodes.  Returns true on success.
 */
using GradFn = std::function<bool(const NodeProto &node, const std::string &output_grad,
                                  std::unordered_map<std::string, std::string> &grad_accum,
                                  int &counter, FunctionProto &func)>;

/**
 * Represents a mapping from op_type strings to their corresponding GradFn implementations.
 */
using GradRegistry = std::unordered_map<std::string, GradFn>;

/**
 * Returns a reference to the built-in gradient registry.
 *
 * The registry contains backward rules for all natively supported operators.
 * Callers who need a mutable copy should copy the returned registry and extend
 * it via RegisterGradientFunction.
 */
const GradRegistry &DefaultGradRegistry();

/**
 * Registers a custom backward function for @p op_type in @p registry.
 *
 * Inserts or replaces the entry for @p op_type.  Pass a copy of
 * DefaultGradRegistry() to extend the built-in set while keeping the defaults.
 *
 * @param op_type  The ONNX operator type name (e.g. "MyCustomOp").
 * @param fn       The backward function implementing the gradient rule.
 * @param registry The registry to insert into.
 */
void RegisterGradientFunction(const std::string &op_type, GradFn fn, GradRegistry &registry);

/**
 * Applies the backward rule for @p node using @p registry.
 *
 * Looks up the output gradient in @p grad_table, then calls the registered
 * backward function and accumulates the resulting input gradients into
 * @p grad_accum.  New nodes are appended to @p func.  @p counter is used to
 * generate unique intermediate names.
 *
 * @throws std::runtime_error if @p node carries an op_type not found in @p registry.
 */
void ApplyBackward(const NodeProto &node,
                   const std::unordered_map<std::string, std::string> &grad_table,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func, const GradRegistry &registry);

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
