// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Registers every built-in ONNX graph-rewriting pattern.
 *
 * Registration is idempotent. Call this function before
 * :cpp:func:`core::builder::CreateRegisteredPatterns` when the standard ONNX
 * pattern set is required.
 */
void RegisterPatterns();

/**
 * Creates one standard ONNX pattern by registered name.
 *
 * Registration is ensured before lookup. When ``priority`` is specified, it
 * overrides the pattern default.
 */
std::unique_ptr<core::builder::PatternOptimization>
CreatePattern(const std::string &name, std::optional<int> priority = std::nullopt);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
