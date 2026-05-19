// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file controlflow/utils.h
 * @brief Declares control-flow shape-inference helpers shared by If, Loop, and Scan.
 *
 * This header exposes utility routines used by control-flow operator schemas to
 * validate axes, clear inferred shapes, and propagate shape/type information.
 */

#pragma once

#include <string>

#include "onnx_lib/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {

void ClearShape(TypeProto &input_type);

int handle_negative_axis_validate(const std::string &attrib, int axis, int rank);

void IfInferenceFunction(InferenceContext &ctx);

void LoopInferenceFunction(InferenceContext &ctx);

void ScanInferenceFunction(InferenceContext &ctx);

} // namespace ONNX_LIGHT_NAMESPACE
