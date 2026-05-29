// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``traditionalml`` op category
// (``ai.onnx.ml`` domain) — exposed so individual cases live in separate
// translation units yet can be invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Binarizer`` backend test node case(s).
void RegisterBinarizerCases(std::vector<TestCase> &registry);

/// Registers the ``ArrayFeatureExtractor`` backend test node case(s).
void RegisterArrayFeatureExtractorCases(std::vector<TestCase> &registry);

/// Registers the ``LabelEncoder`` backend test node case(s).
void RegisterLabelEncoderCases(std::vector<TestCase> &registry);

/// Collects all ``traditionalml`` op category backend test node cases by
/// invoking every ``Register*Cases`` helper declared in this header.
void CollectTraditionalMLTestCases(std::vector<TestCase> &registry,
                                   const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
