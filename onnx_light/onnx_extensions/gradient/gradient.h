// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_extensions/gradient/gradient/grad_dispatcher.h"
#include "onnx_proto/onnx.h"
#include <span>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_gradient {

/**
 * Computes the gradient FunctionProto from a list of ONNX nodes.
 *
 * Performs reverse-mode automatic differentiation over the given nodes and
 * returns a FunctionProto that computes the partial derivatives of @p y with
 * respect to each variable in @p xs.
 *
 * @code
 * X --.
 *     +--> [nodes] --> Y
 * W --'
 *
 * X ---.
 *      |
 * W ---+--> [GradientOfNodes(xs=["X", "W"], y="Y", dy)] --> grad_X, grad_W
 *      |
 * dy --'
 * @endcode
 *
 * The returned FunctionProto has:
 *   - inputs : xs values followed by zs values, then "dy" (the incoming
 *              gradient of y, typically ones_like(y) for a scalar loss).
 *   - outputs: one gradient tensor per element of @p xs, named "grad_<xs[i]>".
 *
 * @param nodes        The forward computation nodes in topological order.
 * @param inputs       Names of all graph inputs.  Accepted for API completeness;
 *                     unused by the current algorithm but reserved for future
 *                     use (e.g. gradient pruning based on graph-input status).
 * @param initializers Constant tensors embedded in the forward graph.
 * @param xs           Variable names to differentiate with respect to.
 * @param y            The output tensor name whose gradient is computed.
 * @param zs           Additional non-differentiable input variable names.
 * @param registry     Operator-to-GradFn map used for backward dispatch.
 *                     Defaults to DefaultGradRegistry().  Pass a copy with
 *                     additional entries (via RegisterGradientFunction) to
 *                     support custom operators.
 * @return             A FunctionProto encoding the gradient computation.
 * @throws std::invalid_argument if @p xs is empty, @p y is empty, or @p y
 *         cannot be reached from the given nodes.
 * @throws std::runtime_error if an op_type is not found in @p registry on the
 *         path from the inputs to @p y.
 */
FunctionProto GradientOfNodes(std::span<const NodeProto> nodes, std::span<const std::string> inputs,
                              std::span<const TensorProto> initializers,
                              std::span<const std::string> xs, const std::string &y,
                              std::span<const std::string> zs,
                              const GradRegistry &registry = DefaultGradRegistry());

/**
 * Computes the gradient FunctionProto from an existing FunctionProto.
 *
 * The @p function is expected to take its initializers as regular inputs (i.e.
 * the caller bakes model parameters into the function's input list rather than
 * embedding them as graph initializers). This is a common pattern when a model
 * is expressed as a pure function for training purposes.
 *
 * @code
 * X ---.
 *      +--> [FunctionProto] --> Y
 * W ---'
 *
 * X ---.
 *      |
 * W ---+--> [GradientOfFunction(xs=["X", "W"], y="Y", dy)] --> grad_X, grad_W
 *      |
 * dy --'
 * @endcode
 *
 * @param function  The forward computation as a FunctionProto.
 * @param xs        Variable names (among @p function inputs) to differentiate
 *                  with respect to.
 * @param y         The output tensor name whose gradient is computed.
 * @param zs        Additional non-differentiable input variable names.
 * @param registry  Operator-to-GradFn map used for backward dispatch.
 *                  Defaults to DefaultGradRegistry().
 * @return          A FunctionProto encoding the gradient computation.
 * @throws std::invalid_argument on invalid arguments (same as GradientOfNodes).
 * @throws std::runtime_error on operators not found in @p registry.
 */
FunctionProto GradientOfFunction(const FunctionProto &function, std::span<const std::string> xs,
                                 const std::string &y, std::span<const std::string> zs,
                                 const GradRegistry &registry = DefaultGradRegistry());

} // namespace ONNX_LIGHT_NAMESPACE::onnx_gradient
