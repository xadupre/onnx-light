// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>
#include <string>

#include "onnx/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace version_conversion {

// Exception thrown when version conversion fails.
struct ConvertError final : public std::runtime_error {
  using std::runtime_error::runtime_error;
  explicit ConvertError(const std::string &message) : std::runtime_error(message) {}
};

#define fail_convert(...)                                                                          \
  ONNX_THROW_EX(ONNX_LIGHT_NAMESPACE::version_conversion::ConvertError(                            \
      ONNX_LIGHT_NAMESPACE::MakeString(__VA_ARGS__)))

} // namespace version_conversion
} // namespace ONNX_LIGHT_NAMESPACE
