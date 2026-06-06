// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/test_case.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``image`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``ai.onnx::ImageDecoder`` backend test node case(s).
void RegisterImageDecoderCases(std::vector<TestCase> &registry);

/// Collects all ``image`` op category backend test node cases by
/// invoking every ``Register*Cases`` helper declared in this header.
void CollectImageTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
