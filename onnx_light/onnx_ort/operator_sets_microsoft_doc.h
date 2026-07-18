// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace microsoft {

/// Returns the documentation string for the BiasGelu operator.
std::string MakeBiasGeluDoc();

/// Returns the documentation string for the BiasGeluGrad_dX operator.
std::string MakeBiasGeluGradDxDoc();

} // namespace microsoft
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
