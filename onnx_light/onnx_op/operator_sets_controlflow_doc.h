// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace controlflow {

/**
 * Returns the documentation string for the If operator.
 *
 * @return Documentation string for the If operator.
 */
std::string MakeIfDoc();

/**
 * Returns the output description string for the If operator at the given
 * opset version.
 *
 * @param since_version Opset version for which to generate the description.
 * @return Output description string.
 */
std::string MakeIfOutputDescription(int since_version);

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
