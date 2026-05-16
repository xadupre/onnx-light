// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file operator_sets_training.h
 * @brief Declares ai.onnx.training operator-set schema registration helpers.
 *
 * This header defines the training-domain opset container and registration
 * entry point used to register ai.onnx.training schemas.
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {

// Declare training operators.

// Iterate over schema from ai.onnx.training version 1
class OpSet_OnnxTraining_ver1 {
public:
  static void ForEachSchema(const std::function<void(OpSchema &&)> & /* fn */) {}
};

// Register training operators.
ONNX_API inline void RegisterOnnxTrainingOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_OnnxTraining_ver1>();
}

} // namespace ONNX_LIGHT_NAMESPACE
