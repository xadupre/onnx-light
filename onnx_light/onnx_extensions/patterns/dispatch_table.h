// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

/**
 * Registers every built-in ONNX graph-rewriting pattern.
 *
 * Registration is idempotent. Call this function before
 * :cpp:func:`core::builder::CreateRegisteredPatterns` when the standard ONNX
 * pattern set is required.
 */
void RegisterPatterns();

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
