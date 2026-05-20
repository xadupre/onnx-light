// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {

std::string MakeBinaryLogicalOperatorDoc(const char *op_type, int since_version);
std::string MakeNotLogicalOperatorDoc();

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
