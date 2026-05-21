// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "onnx_op/light_op_schema.h"
#include "onnx_op/operator_sets_controlflow.h"
#include "onnx_op/operator_sets_generator.h"
#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_math.h"
#include "onnx_op/operator_sets_reduction.h"
#include "onnx_op/operator_sets_sequence.h"
#include "onnx_op/operator_sets_tensor.h"
#include "onnx_op/operator_sets_traditionalml.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

std::vector<LightOpSchema> GetAllOnnxOpSchemasWithHistory();

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
