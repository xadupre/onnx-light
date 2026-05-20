// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "onnx_op/light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace reduction {

std::string MakeReduceSumDoc(int since_version);

} // namespace reduction
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
