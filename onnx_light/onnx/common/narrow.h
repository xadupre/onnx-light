#pragma once

#include "onnx/common/common.h"

#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE {

template <class T, class U> constexpr T narrow(U value) {
  static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
  static_assert(std::is_arithmetic_v<U>, "U must be arithmetic");

  const T result = static_cast<T>(value);

  if (static_cast<U>(result) != value ||
      (std::is_signed_v<T> != std::is_signed_v<U> && ((result < T{}) != (value < U{})))) {
    throw std::runtime_error("narrow: value out of range");
  }

  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE
