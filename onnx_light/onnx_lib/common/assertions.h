// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx_pb.h"
#include "string_utils.h"

#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {

/**
 * @brief Exception type thrown when an ONNX assertion fails.
 *
 * Derives from @c std::runtime_error and carries the assertion message.
 */
struct assert_error : public std::runtime_error {
public:
  /**
   * @brief Constructs an assert_error with the given message.
   * @param msg Human-readable description of the failed assertion.
   */
  explicit assert_error(const std::string &msg) : runtime_error(msg) {}
};

/**
 * @brief Exception type thrown when a tensor-specific assertion fails.
 *
 * Specialization of assert_error used by TENSOR_ASSERTM to distinguish
 * tensor-related failures from general assertion failures.
 */
struct tensor_error : public assert_error {
public:
  /**
   * @brief Constructs a tensor_error with the given message.
   * @param msg Human-readable description of the failed tensor assertion.
   */
  explicit tensor_error(const std::string &msg) : assert_error(msg) {}
};

/**
 * @brief Throws an assert_error with the given message.
 *
 * When exceptions are disabled (ONNX_NO_EXCEPTIONS), prints the message to
 * @c std::cerr and calls @c std::abort() instead.
 *
 * @param msg Error message to embed in the exception.
 */
[[noreturn]] void throw_assert_error(const std::string &msg);

/**
 * @brief Throws a tensor_error with the given message.
 *
 * When exceptions are disabled (ONNX_NO_EXCEPTIONS), prints the message to
 * @c std::cerr and calls @c std::abort() instead.
 *
 * @param msg Error message to embed in the exception.
 */
[[noreturn]] void throw_tensor_error(const std::string &msg);

} // namespace ONNX_LIGHT_NAMESPACE

/**
 * @def _ONNX_EXPECT(x, y)
 * @brief Provides a branch-prediction hint to the compiler.
 *
 * On GCC, Clang, and ICC expands to @c __builtin_expect((x),(y)) so the
 * compiler can optimize the fast path.  On all other compilers the macro is a
 * no-op that simply evaluates to @p x.
 *
 * @param x The expression whose value is being predicted.
 * @param y The expected value of @p x (0 for unlikely, 1 for likely).
 */
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define _ONNX_EXPECT(x, y) (__builtin_expect((x), (y)))
#else
#define _ONNX_EXPECT(x, y) (x)
#endif

// The message arguments are concatenated with std::stringstream (via MakeString),
// so std::string, std::string_view, and numbers are all supported directly without
// format specifiers or .c_str().

/**
 * @def ONNX_ASSERT(cond)
 * @brief Asserts that @p cond is true; throws assert_error if it is not.
 *
 * The failure message includes the source file, line number, enclosing
 * function name, and the stringified condition.  The branch is annotated as
 * unlikely via _ONNX_EXPECT so the fast path incurs no extra overhead.
 *
 * @param cond Boolean expression that is expected to be true.
 */
#define ONNX_ASSERT(cond)                                                                          \
  if (_ONNX_EXPECT(!(cond), 0)) { /* NOLINT(readability-simplify-boolean-expr) */                  \
    std::string error_msg = ::ONNX_LIGHT_NAMESPACE::MakeString(                                    \
        __FILE__, ":", __LINE__, ": ", __func__, ": Assertion `", #cond, "` failed.");             \
    throw_assert_error(error_msg);                                                                 \
  }

/**
 * @def ONNX_ASSERTM(cond, ...)
 * @brief Asserts that @p cond is true; throws assert_error with a custom message if not.
 *
 * The message arguments are concatenated via MakeString (using std::stringstream),
 * so std::string, std::string_view, and numeric types are all accepted directly
 * without format specifiers or .c_str().  The resulting exception message has the form:
 * @code
 * file:line: func: Assertion `cond` failed: <message>
 * @endcode
 *
 * @param cond Boolean expression that is expected to be true.
 * @param ...  Arguments concatenated by MakeString to describe the failure.
 */
#define ONNX_ASSERTM(cond, ...)                                                                    \
  /* NOLINTNEXTLINE */                                                                             \
  if (_ONNX_EXPECT(!(cond), 0)) {                                                                  \
    std::string error_msg =                                                                        \
        ::ONNX_LIGHT_NAMESPACE::MakeString(__FILE__, ":", __LINE__, ": ", __func__,                \
                                           ": Assertion `", #cond, "` failed: ", __VA_ARGS__);     \
    throw_assert_error(error_msg);                                                                 \
  }

/**
 * @def TENSOR_ASSERTM(cond, ...)
 * @brief Asserts that @p cond is true; throws tensor_error with a custom message if not.
 *
 * Identical in structure to ONNX_ASSERTM but throws tensor_error instead of
 * assert_error, allowing callers to catch tensor-specific failures separately.
 *
 * @param cond Boolean expression that is expected to be true.
 * @param ...  Arguments concatenated by MakeString to describe the failure.
 */
#define TENSOR_ASSERTM(cond, ...)                                                                  \
  if (_ONNX_EXPECT(!(cond), 0)) {                                                                  \
    std::string error_msg =                                                                        \
        ::ONNX_LIGHT_NAMESPACE::MakeString(__FILE__, ":", __LINE__, ": ", __func__,                \
                                           ": Assertion `", #cond, "` failed: ", __VA_ARGS__);     \
    throw_tensor_error(error_msg);                                                                 \
  }
