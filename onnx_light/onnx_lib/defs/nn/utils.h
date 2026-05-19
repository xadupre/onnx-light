// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file nn/utils.h
 * @brief Declares shared neural-network operator helpers.
 *
 * This header provides utility APIs used by NN operator schemas, including
 * Conv/Pool stride extraction, Attention type propagation, and function-body
 * construction helpers.
 */

#pragma once

#include <vector>

#include "onnx_lib/common/assertions.h"
#include "onnx_lib/defs/function.h"

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Reads and validates the 'strides' attribute for Conv/Pool shape inference.
 * Returns the attribute value or a default value if the attribute is not present.
 */
std::vector<int64_t> getConvPoolStrides(InferenceContext &ctx, size_t n_input_dims);

/** Implements shape and type propagation for Attention (23-). */
void AttentionPropagateElemTypeFromInputToOutput(InferenceContext &ctx);

/** Implements CausalMask for Attention. */
bool AttentionAppendFunctionCausalMask(const FunctionBodyBuildContext &ctx,
                                       FunctionBuilder &builder, bool padding);

} // namespace ONNX_LIGHT_NAMESPACE
