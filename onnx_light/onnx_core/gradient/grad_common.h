// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_proto/onnx.h"
#include <string>
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE::core::gradient {

/**
 * Accumulates @p contrib_name into @p acc_name inside @p func.
 *
 * If @p acc_name is empty, assigns it to @p contrib_name directly.
 * Otherwise inserts an Add node and updates @p acc_name to the result.
 */
void AccumulateGrad(const std::string &contrib_name, std::string &acc_name, int &counter,
                    FunctionProto &func);

/**
 * Returns a new unique intermediate name composed of @p prefix and @p counter.
 *
 * Increments @p counter.
 */
std::string NewGradName(const std::string &prefix, int &counter);

} // namespace ONNX_LIGHT_NAMESPACE::core::gradient
