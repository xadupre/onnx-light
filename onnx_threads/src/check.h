// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sstream>
#include <stdexcept>

namespace onnx_threads::detail {

template <typename Exception, typename... Args> void Check(bool condition, const Args &...args) {
  if (!condition) {
    std::ostringstream message;
    (message << ... << args);
    throw Exception(message.str());
  }
}

template <typename... Args> void Enforce(bool condition, const Args &...args) {
  Check<std::runtime_error>(condition, args...);
}

template <typename... Args> void EnforceInvalid(bool condition, const Args &...args) {
  Check<std::invalid_argument>(condition, args...);
}

} // namespace onnx_threads::detail
