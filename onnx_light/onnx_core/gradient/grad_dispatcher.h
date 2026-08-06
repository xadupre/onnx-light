// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::gradient {

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

/** Hash functor for std::pair<std::string, std::string> registry keys.
 *  Uses a FNV-inspired mixing to combine the two component hashes. */
struct PairStringHash {
  std::size_t operator()(const std::pair<std::string, std::string> &p) const noexcept {
    std::size_t h1 = std::hash<std::string>{}(p.first);
    std::size_t h2 = std::hash<std::string>{}(p.second);
    // FNV-inspired mixing: rotate h1 and combine with h2 via golden-ratio multiply.
    return h1 ^ (h2 * 2654435761ULL + 0x9e3779b9ULL + (h1 << 6) + (h1 >> 2));
  }
};

/**
 * Represents a mapping from (domain, op_type) pairs to their corresponding GradFn
 * implementations.  The empty string "" denotes the default ONNX operator domain.
 */
using GradRegistry =
    std::unordered_map<std::pair<std::string, std::string>, GradFn, PairStringHash>;

/**
 * Returns a reference to the built-in gradient registry.
 *
 * The registry contains backward rules for all natively supported operators.
 * Callers who need a mutable copy should copy the returned registry and extend
 * it via RegisterGradientFunction.
 */
const GradRegistry &DefaultGradRegistry();

/**
 * Registers a custom backward function for (@p domain, @p op_type) in @p registry.
 *
 * Inserts or replaces the entry for the given key.  Pass a copy of
 * DefaultGradRegistry() to extend the built-in set while keeping the defaults.
 * Use an empty string for @p domain to denote the default ONNX operator domain.
 *
 * @param domain   The operator domain (e.g. "" for standard ONNX, "com.example" for custom).
 * @param op_type  The ONNX operator type name (e.g. "MyCustomOp").
 * @param fn       The backward function implementing the gradient rule.
 * @param registry The registry to insert into.
 */
void RegisterGradientFunction(const std::string &domain, const std::string &op_type, GradFn fn,
                              GradRegistry &registry);

/**
 * Applies the backward rule for @p node using @p registry.
 *
 * Looks up the output gradient in @p grad_table, then calls the registered
 * backward function and accumulates the resulting input gradients into
 * @p grad_accum.  New nodes are appended to @p func.  @p counter is used to
 * generate unique intermediate names.
 *
 * Raises an exception if the (domain, op_type) of @p node is not found in
 * @p registry, as the whole gradient computation would be incorrect.
 */
void ApplyBackward(const NodeProto &node,
                   const std::unordered_map<std::string, std::string> &grad_table,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func, const GradRegistry &registry);

} // namespace ONNX_LIGHT_NAMESPACE::core::gradient
