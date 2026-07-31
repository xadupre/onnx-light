// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE {

/**
 * @brief Multiplies two non-negative integers, reporting overflow.
 *
 * Uses @c __builtin_mul_overflow on GCC/Clang and a manual check on MSVC.
 * Precondition: @p a and @p b must be non-negative. The MSVC fallback only
 * handles non-negative inputs correctly; passing a negative value can silently
 * overflow. safe_dim_product() enforces this by checking each dim before
 * calling, and the accumulated result stays non-negative because it aborts on
 * overflow.
 *
 * @param a First non-negative operand.
 * @param b Second non-negative operand.
 * @param result Receives the product when no overflow occurs.
 * @returns True on overflow, false otherwise.
 */
inline bool checked_mul_overflow(int64_t a, int64_t b, int64_t *result) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_mul_overflow(a, b, result);
#else
  assert(a >= 0 && b >= 0 && "checked_mul_overflow requires non-negative inputs on MSVC");
  if (a > 0 && b > std::numeric_limits<int64_t>::max() / a) {
    return true;
  }
  *result = a * b;
  return false;
#endif
}

/**
 * @brief Computes the product of dims over an iterator range with overflow checks.
 *
 * Calls @p on_error with a @c const @c char* message on a negative dim or on
 * overflow. @p on_error must not return (i.e. must throw or abort).
 *
 * @param begin Iterator to the first dimension.
 * @param end Iterator past the last dimension.
 * @param on_error Handler invoked on a negative dimension or overflow.
 * @returns The product of the dimensions in the range.
 */
template <typename Iter, typename ErrorHandler>
[[nodiscard]] inline int64_t safe_dim_product(Iter begin, Iter end, ErrorHandler on_error) {
  int64_t result = 1;
  for (auto it = begin; it != end; ++it) {
    auto dim = static_cast<int64_t>(*it);
    if (dim < 0) {
      on_error("Negative dimension value");
      return result; // unreachable if on_error throws; guards against misuse
    }
    if (checked_mul_overflow(result, dim, &result)) {
      on_error("Tensor dimension product overflow");
      return result;
    }
  }
  return result;
}

/**
 * @brief Computes the product of dims in a container with overflow checks.
 *
 * Delegates to the iterator-pair overload of safe_dim_product().
 *
 * @param dims Container of dimensions.
 * @param on_error Handler invoked on a negative dimension or overflow.
 * @returns The product of the dimensions in the container.
 */
template <typename DimsContainer, typename ErrorHandler>
[[nodiscard]] inline int64_t safe_dim_product(const DimsContainer &dims, ErrorHandler on_error) {
  return safe_dim_product(std::begin(dims), std::end(dims), on_error);
}

/**
 * @brief Casts an int64_t to size_t, reporting values out of range.
 *
 * Calls @p on_error if the value exceeds the size_t range (relevant for 32-bit
 * platforms where size_t is 32 bits). @p value must be non-negative (callers
 * ensure this via prior overflow checks).
 *
 * @param value Non-negative value to cast.
 * @param on_error Handler invoked when @p value is too large for size_t.
 * @returns The value cast to size_t.
 */
template <typename ErrorHandler>
[[nodiscard]] inline size_t safe_cast_to_size(int64_t value, ErrorHandler on_error) {
  if (static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max()) {
    on_error("Value too large for this platform");
  }
  return static_cast<size_t>(value);
}

} // namespace ONNX_LIGHT_NAMESPACE
