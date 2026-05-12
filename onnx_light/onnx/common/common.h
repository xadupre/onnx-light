// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file common.h
 * @brief Portability macros for exception handling and class rule-of-five enforcement.
 *
 * Provides two groups of macros:
 *  - Exception-handling macros (@ref ONNX_THROW, @ref ONNX_THROW_EX, @ref ONNX_TRY,
 *    @ref ONNX_CATCH, @ref ONNX_HANDLE_EXCEPTION) that transparently switch between
 *    C++ exceptions and abort()-based termination depending on whether
 *    @c ONNX_NO_EXCEPTIONS is defined.
 *  - Copy/move-deletion macros (@ref ONNX_DISALLOW_COPY, @ref ONNX_DISALLOW_ASSIGNMENT,
 *    @ref ONNX_DISALLOW_MOVE, @ref ONNX_DISALLOW_COPY_AND_ASSIGNMENT,
 *    @ref ONNX_DISALLOW_COPY_ASSIGNMENT_AND_MOVE) that delete the corresponding
 *    special-member functions in a class body.
 */

#pragma once

#include "string_utils.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

/**
 * @def ONNX_UNUSED_PARAMETER(x)
 * @brief Suppresses compiler warnings about an unreferenced function parameter.
 *
 * Casts @p x to @c void so that compilers do not emit an unused-parameter
 * diagnostic.  Use this in function bodies where a parameter is intentionally
 * left unused (e.g. in a no-op override or a debug-only code path).
 *
 * @param x The parameter name to mark as intentionally unused.
 */
#define ONNX_UNUSED_PARAMETER(x) (void)(x)

#ifdef ONNX_NO_EXCEPTIONS
/**
 * @def ONNX_THROW(...)
 * @brief Terminates the process with a formatted error message.
 *
 * When @c ONNX_NO_EXCEPTIONS is defined, prints the message produced by
 * @c MakeString(__VA_ARGS__) to @c std::cerr and calls @c std::abort().
 * When @c ONNX_NO_EXCEPTIONS is **not** defined, throws a
 * @c std::runtime_error whose message is produced by
 * @c MakeString(__VA_ARGS__).
 *
 * @param ... Arguments forwarded to @c MakeString to build the error message.
 */
#define ONNX_THROW(...)                                                                            \
  do {                                                                                             \
    std::cerr << ONNX_LIGHT_NAMESPACE::MakeString(__VA_ARGS__);                                    \
    std::abort();                                                                                  \
  } while (false)

/**
 * @def ONNX_THROW_EX(ex)
 * @brief Terminates the process by printing an existing exception's message.
 *
 * When @c ONNX_NO_EXCEPTIONS is defined, prints @c ex.what() to @c std::cerr
 * and calls @c std::abort().  When @c ONNX_NO_EXCEPTIONS is **not** defined,
 * re-throws @p ex using @c throw.
 *
 * @param ex An exception object whose @c what() method provides the message.
 */
#define ONNX_THROW_EX(ex)                                                                          \
  do {                                                                                             \
    std::cerr << ex.what() << std::endl; /* NOLINT */                                              \
    std::abort();                                                                                  \
  } while (false)

/**
 * @def ONNX_TRY
 * @brief Opens a guarded block; equivalent to @c try when exceptions are enabled.
 *
 * When @c ONNX_NO_EXCEPTIONS is defined, expands to @c if(true) so the
 * guarded code still runs without any exception-handling overhead.
 * When @c ONNX_NO_EXCEPTIONS is **not** defined, expands to @c try.
 *
 * Use together with @ref ONNX_CATCH and @ref ONNX_HANDLE_EXCEPTION.
 */
#define ONNX_TRY if (true)

/**
 * @def ONNX_CATCH(x)
 * @brief Introduces a catch clause; a no-op when exceptions are disabled.
 *
 * When @c ONNX_NO_EXCEPTIONS is defined, expands to @c else @c if(false)
 * so the catch body is compiled but never executed.
 * When @c ONNX_NO_EXCEPTIONS is **not** defined, expands to @c catch(x).
 *
 * @param x The exception type (or @c ...) to catch.
 */
#define ONNX_CATCH(x) else if (false)

/**
 * @def ONNX_HANDLE_EXCEPTION(func)
 * @brief Invokes an exception-handling callback; a no-op when exceptions are disabled.
 *
 * When @c ONNX_NO_EXCEPTIONS is defined, expands to nothing so the call is
 * omitted entirely.  When @c ONNX_NO_EXCEPTIONS is **not** defined, expands
 * to @c func() to invoke the provided callable.
 *
 * @param func A zero-argument callable invoked inside a catch block to perform
 *             additional exception handling.
 */
#define ONNX_HANDLE_EXCEPTION(func)

#else
/// @copydoc ONNX_THROW
#define ONNX_THROW(...) throw std::runtime_error(ONNX_LIGHT_NAMESPACE::MakeString(__VA_ARGS__))
/// @copydoc ONNX_THROW_EX
#define ONNX_THROW_EX(ex) throw ex

/// @copydoc ONNX_TRY
#define ONNX_TRY try
/// @copydoc ONNX_CATCH
#define ONNX_CATCH(x) catch (x)
/// @copydoc ONNX_HANDLE_EXCEPTION
#define ONNX_HANDLE_EXCEPTION(func) func()
#endif

/**
 * @def ONNX_DISALLOW_COPY(TypeName)
 * @brief Deletes the copy constructor of @p TypeName.
 *
 * Place inside a class body to prevent copy construction.
 * Typically used together with @ref ONNX_DISALLOW_ASSIGNMENT.
 *
 * @param TypeName The unqualified name of the enclosing class.
 */
#define ONNX_DISALLOW_COPY(TypeName) TypeName(const TypeName &) = delete

/**
 * @def ONNX_DISALLOW_ASSIGNMENT(TypeName)
 * @brief Deletes the copy-assignment operator of @p TypeName.
 *
 * Place inside a class body to prevent copy assignment.
 * Typically used together with @ref ONNX_DISALLOW_COPY.
 *
 * @param TypeName The unqualified name of the enclosing class.
 */
#define ONNX_DISALLOW_ASSIGNMENT(TypeName) TypeName &operator=(const TypeName &) = delete

/**
 * @def ONNX_DISALLOW_COPY_AND_ASSIGNMENT(TypeName)
 * @brief Deletes both the copy constructor and copy-assignment operator of @p TypeName.
 *
 * Expands to @ref ONNX_DISALLOW_COPY followed by @ref ONNX_DISALLOW_ASSIGNMENT,
 * making the class non-copyable.
 *
 * @param TypeName The unqualified name of the enclosing class.
 */
#define ONNX_DISALLOW_COPY_AND_ASSIGNMENT(TypeName)                                                \
  ONNX_DISALLOW_COPY(TypeName);                                                                    \
  ONNX_DISALLOW_ASSIGNMENT(TypeName)

/**
 * @def ONNX_DISALLOW_MOVE(TypeName)
 * @brief Deletes both the move constructor and move-assignment operator of @p TypeName.
 *
 * Place inside a class body to prevent move construction and move assignment,
 * making the class immovable.
 *
 * @param TypeName The unqualified name of the enclosing class.
 */
#define ONNX_DISALLOW_MOVE(TypeName)                                                               \
  TypeName(TypeName &&) = delete;                                                                  \
  TypeName &operator=(TypeName &&) = delete

/**
 * @def ONNX_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName)
 * @brief Deletes all copy and move special-member functions of @p TypeName.
 *
 * Expands to @ref ONNX_DISALLOW_COPY_AND_ASSIGNMENT followed by
 * @ref ONNX_DISALLOW_MOVE, making the class neither copyable nor movable.
 *
 * @param TypeName The unqualified name of the enclosing class.
 */
#define ONNX_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName)                                           \
  ONNX_DISALLOW_COPY_AND_ASSIGNMENT(TypeName);                                                     \
  ONNX_DISALLOW_MOVE(TypeName)
