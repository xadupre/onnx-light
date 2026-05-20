// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {
namespace detail {

std::string BuildLogicalOperatorDoc(const char *name, int since_version);

} // namespace detail
} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
