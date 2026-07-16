// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_gradient {

/**
 * Computes the gradient FunctionProto from a list of ONNX nodes.
 *
 * Performs reverse-mode automatic differentiation over the given nodes and
 * returns a FunctionProto that computes the partial derivatives of @p y with
 * respect to each variable in @p xs.
 *
 * The returned FunctionProto has:
 *   - inputs : xs values followed by zs values, then "dy" (the incoming
 *              gradient of y, typically ones_like(y) for a scalar loss).
 *   - outputs: one gradient tensor per element of @p xs, named "grad_<xs[i]>".
 *
 * Supported forward operators: MatMul, Gemm, Add, Sub, Mul, Div, Neg, Identity,
 * Relu, ReduceSum, ReduceMean, Reshape, Transpose, Sigmoid, Tanh.
 *
 * @param nodes        The forward computation nodes in topological order.
 * @param inputs       Names of all graph inputs.  Accepted for API completeness;
 *                     unused by the current algorithm but reserved for future
 *                     use (e.g. gradient pruning based on graph-input status).
 * @param initializers Constant tensors embedded in the forward graph.
 * @param xs           Variable names to differentiate with respect to.
 * @param y            The output tensor name whose gradient is computed.
 * @param zs           Additional non-differentiable input variable names.
 * @return             A FunctionProto encoding the gradient computation.
 * @throws std::invalid_argument if @p xs is empty, @p y is empty, or @p y
 *         cannot be reached from the given nodes.
 * @throws std::runtime_error if an unsupported op_type is encountered on the
 *         path from the inputs to @p y.
 */
FunctionProto GradientOfNodes(const std::vector<NodeProto> &nodes,
                              const std::vector<std::string> &inputs,
                              const std::vector<TensorProto> &initializers,
                              const std::vector<std::string> &xs, const std::string &y,
                              const std::vector<std::string> &zs);

/**
 * Computes the gradient FunctionProto from an existing FunctionProto.
 *
 * The @p function is expected to take its initializers as regular inputs (i.e.
 * the caller bakes model parameters into the function's input list rather than
 * embedding them as graph initializers). This is a common pattern when a model
 * is expressed as a pure function for training purposes.
 *
 * @param function  The forward computation as a FunctionProto.
 * @param xs        Variable names (among @p function inputs) to differentiate
 *                  with respect to.
 * @param y         The output tensor name whose gradient is computed.
 * @param zs        Additional non-differentiable input variable names.
 * @return          A FunctionProto encoding the gradient computation.
 * @throws std::invalid_argument on invalid arguments (same as GradientOfNodes).
 * @throws std::runtime_error on unsupported operators.
 */
FunctionProto GradientOfFunction(const FunctionProto &function, const std::vector<std::string> &xs,
                                 const std::string &y, const std::vector<std::string> &zs);

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
