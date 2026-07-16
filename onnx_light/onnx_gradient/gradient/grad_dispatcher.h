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
 * Applies the backward rule for @p node, dispatching to the per-op implementation.
 *
 * Looks up the output gradient in @p grad_table, then calls the appropriate
 * per-operator backward function and accumulates the resulting input gradients
 * into @p grad_accum.  New nodes are appended to @p func.  @p counter is used
 * to generate unique intermediate names.
 *
 * @throws std::runtime_error if @p node carries an unsupported op_type.
 */
void ApplyBackward(const NodeProto &node,
                   const std::unordered_map<std::string, std::string> &grad_table,
                   std::unordered_map<std::string, std::string> &grad_accum, int &counter,
                   FunctionProto &func);

} // namespace onnx_gradient
} // namespace ONNX_LIGHT_NAMESPACE
