// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``text`` op category — exposed
// so individual cases live in separate translation units yet can be invoked
// from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``StringConcat`` backend test node case(s).
void RegisterStringConcatCases(std::vector<TestCase> &registry);

/// Registers the ``StringSplit`` backend test node case(s).
void RegisterStringSplitCases(std::vector<TestCase> &registry);

/// Registers the ``StringNormalizer`` backend test node case(s).
void RegisterStringNormalizerCases(std::vector<TestCase> &registry);

/// Registers the ``RegexFullMatch`` backend test node case(s).
void RegisterRegexFullMatchCases(std::vector<TestCase> &registry);

/// Registers the ``TfIdfVectorizer`` backend test node case(s).
void RegisterTfIdfVectorizerCases(std::vector<TestCase> &registry);

/// Collects all ``text`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectTextTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
