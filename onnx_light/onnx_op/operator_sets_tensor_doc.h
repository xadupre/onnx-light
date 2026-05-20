// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

std::string MakeCastDoc(int since_version);
std::string MakeCastInputTypeConstraintDescription(int since_version);
std::string MakeCastOutputTypeConstraintDescription(int since_version);

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
