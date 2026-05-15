// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file errors.h
 * @brief Exception types and helper macros for ONNX version conversion errors.
 *
 * Provides the ConvertError exception class and the fail_convert() macro used
 * throughout the version_converter subsystem to report unsupported or invalid
 * version-conversion operations.
 */

#pragma once

#include <stdexcept>
#include <string>

#include "onnx/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace version_conversion {

/**
 * @brief Exception thrown when an ONNX operator version conversion fails.
 *
 * Derives from @c std::runtime_error.  Raise this exception (typically via
 * the fail_convert() macro) to signal that the requested version conversion
 * is not supported or that the model is in an invalid state for conversion.
 *
 * ### Example
 * @code{.cpp}
 * if (!adapter_found) {
 *   fail_convert("No adapter for op ", op_name, " from version ", from, " to ", to);
 * }
 * @endcode
 */
struct ConvertError final : public std::runtime_error {
  using std::runtime_error::runtime_error;

  /**
   * @brief Constructs a ConvertError with the given message.
   * @param message Human-readable description of the conversion failure.
   */
  explicit ConvertError(const std::string &message) : std::runtime_error(message) {}
};

/**
 * @def fail_convert(...)
 * @brief Throws a ConvertError whose message is built from the given arguments.
 *
 * Concatenates all arguments via MakeString() and throws a ConvertError with
 * the resulting string.  When @c ONNX_NO_EXCEPTIONS is defined the macro
 * prints the message to @c std::cerr and calls @c std::abort() instead.
 *
 * @param ... One or more values that can be streamed into a @c std::string via
 *            MakeString().  Common examples are string literals, operator
 *            names, and version numbers.
 *
 * @throws ConvertError Always (unless exceptions are disabled).
 */
#define fail_convert(...)                                                                          \
  ONNX_THROW_EX(ONNX_LIGHT_NAMESPACE::version_conversion::ConvertError(                            \
      ONNX_LIGHT_NAMESPACE::MakeString(__VA_ARGS__)))

} // namespace version_conversion
} // namespace ONNX_LIGHT_NAMESPACE
