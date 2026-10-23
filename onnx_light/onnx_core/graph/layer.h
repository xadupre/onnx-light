// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file layer.h
 * @brief Validates explicit graph boundaries for ai.rt::Layer.
 */

#include "onnx_proto/onnx.h"

#include <functional>

namespace ONNX_LIGHT_NAMESPACE::core::graph {

using LayerFunctionLookupFn = std::function<bool(const NodeProto &)>;

/**
 * Validates a Layer's explicit boundary and recursively closed body dependencies.
 *
 * Caller values override body input initializer defaults. Every formal input still
 * requires an explicit positional caller input, even when it has a default.
 */
const GraphProto &ValidateLayer(const NodeProto &node,
                                const LayerFunctionLookupFn &has_local_function = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
