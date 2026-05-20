// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace controlflow {

std::string MakeIfDoc();
std::string MakeIfOutputDescription();
std::string MakeIfValueTypeConstraintDescription(int since_version);

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
