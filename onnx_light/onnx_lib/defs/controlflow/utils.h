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

/**
 * Validates the Scan input/output counts against the declared
 * ``num_scan_inputs`` and returns the number of loop-state variables.
 *
 * Guards both subtractions against ``size_t`` underflow
 * (GHSA-qrhj-v62m-vmpf): when ``num_scan_inputs`` exceeds the input count the
 * subtraction ``num_inputs - num_scan_inputs`` wraps around; when the resulting
 * loop-state-variable count exceeds the output count the second subtraction
 * wraps around.  Both conditions call ``fail_shape_inference`` instead of
 * producing a huge index.
 *
 * Used by ``ScanInferenceFunction`` (opset 9+) and
 * ``ScanInferenceFunction_opset9``.  Opset 8 has an extra
 * ``sequence_lens`` offset and is handled inline in
 * ``ScanInferenceFunction_opset8``.
 */
size_t ValidateScanCountsAndGetNumLoopStateVars(size_t num_inputs, size_t num_scan_inputs,
                                                size_t num_outputs);

void IfInferenceFunction(InferenceContext &ctx);

void LoopInferenceFunction(InferenceContext &ctx);

void ScanInferenceFunction(InferenceContext &ctx);

} // namespace ONNX_LIGHT_NAMESPACE
