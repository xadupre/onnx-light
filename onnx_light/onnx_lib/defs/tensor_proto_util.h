// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file tensor_proto_util.h
 * @brief Compatibility stub that redirects to onnx/defs/tensor_util.h.
 *
 * Upstream ONNX code and vendored headers that include
 * `"onnx/defs/tensor_proto_util.h"` are transparently served by this file.
 * All tensor-parsing and tensor-creation utilities (`ParseData`, `ToTensor`)
 * are defined in onnx/defs/tensor_util.h, which this header re-exports.
 *
 * @see onnx/defs/tensor_util.h
 */

#pragma once

#include "onnx_lib/defs/tensor_util.h"
