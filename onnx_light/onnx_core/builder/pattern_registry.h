// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_core/builder/pattern_optimization.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

/// Factory creating one stateless graph-rewriting pattern.
using PatternFactory = std::function<std::unique_ptr<PatternOptimization>()>;

/// Reports an invalid or duplicate pattern registration.
class PatternRegistrationError : public std::runtime_error {
public:
  /// Constructs an error with the given diagnostic message.
  explicit PatternRegistrationError(const std::string &message) : std::runtime_error(message) {}
};

/**
 * Registers a pattern factory under a stable diagnostic name.
 *
 * Registration order determines the order returned by
 * :cpp:func:`CreateRegisteredPatterns`. Names must be non-empty and unique.
 */
void RegisterPattern(const std::string &name, PatternFactory factory);

/// Returns the registered pattern names in registration order.
std::vector<std::string> RegisteredPatternNames();

/// Creates one pattern instance from every registered factory.
std::vector<std::unique_ptr<PatternOptimization>> CreateRegisteredPatterns();

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
