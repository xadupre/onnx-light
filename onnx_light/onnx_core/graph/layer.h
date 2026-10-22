// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file layer.h
 * @brief Validates explicit graph boundaries for ai.rt::Layer.
 */

#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE::core::graph {

/**
 * Validates a Layer's explicit boundary and recursively closed body dependencies.
 *
 * Caller values override body input initializer defaults. Every formal input still
 * requires an explicit positional caller input, even when it has a default.
 */
const GraphProto &ValidateLayer(const NodeProto &node);

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
