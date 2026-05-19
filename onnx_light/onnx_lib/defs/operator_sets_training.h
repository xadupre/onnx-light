// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file operator_sets_training.h
 * @brief Declares ai.onnx.preview.training operator-set schema registration helpers.
 *
 * This header defines the preview-training-domain opset container and
 * registration entry point used to register ai.onnx.preview.training schemas.
 */

#pragma once

#include "onnx_lib/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {

// Declare preview training operators.
class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Gradient);
class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Momentum);
class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adagrad);
class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adam);

// Iterate over schema from ai.onnx.preview.training version 1
class OpSet_OnnxTraining_ver1 {
public:
  static void ForEachSchema(const std::function<void(OpSchema &&)> &fn) {
    fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Gradient)>());
    fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Momentum)>());
    fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adagrad)>());
    fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adam)>());
  }
};

// Register preview training operators.
ONNX_API inline void RegisterOnnxTrainingOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_OnnxTraining_ver1>();
}

} // namespace ONNX_LIGHT_NAMESPACE
