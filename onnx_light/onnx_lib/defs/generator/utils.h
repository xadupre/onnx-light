// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file generator/utils.h
 * @brief Declares shape-inference helpers for generator-style operators.
 *
 * This header currently provides the Constant operator inference helper used by
 * schema definitions in the generator domain.
 */

#pragma once

#include "onnx_lib/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {

void ConstantOpInference(InferenceContext &ctx);

} // namespace ONNX_LIGHT_NAMESPACE
