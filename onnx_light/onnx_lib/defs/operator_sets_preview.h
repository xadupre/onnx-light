// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file operator_sets_preview.h
 * @brief Declares ai.onnx.preview operator-set schema declarations and registrars.
 *
 * This header provides preview-domain schema declarations and a registration
 * helper for the ai.onnx.preview opset.
 */

#pragma once

#include "onnx_lib/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {

// Declare preview operators.

class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, FlexAttention);

// Iterate over schema from ai.onnx.preview* version 1
class OpSet_OnnxPreview_ver1 {
public:
  static void ForEachSchema(const std::function<void(OpSchema &&)> &fn) {
    fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, FlexAttention)>());
  }
};

// Register preview operators.
ONNX_API inline void RegisterOnnxPreviewOperatorSetSchema() {
  // Preview operators should have only one version.
  // If changes are needed for a specific preview operator,
  // its spec should be modified without increasing its version.
  RegisterOpSetSchema<OpSet_OnnxPreview_ver1>();
}

} // namespace ONNX_LIGHT_NAMESPACE
