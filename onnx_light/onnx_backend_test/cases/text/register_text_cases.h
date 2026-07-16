// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

// Internal (library-private) header: the per-operator ``Register*`` backend
// test registration helpers. These are only ever invoked by the ``Collect*``
// aggregators inside ``lib_onnx_backend_test`` and are compiled with hidden
// visibility, so they are not exported from the shared library. This header is
// pulled in by the matching public header only when ONNX_LIGHT_BACKEND_TEST_INTERNAL is
// defined (i.e. while building the library itself).

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``text`` op category — exposed
// so individual cases live in separate translation units yet can be invoked
// from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``StringConcat`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterStringConcatCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

/// Registers the ``StringSplit`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterStringSplitCases(std::vector<TestCase> &registry,
                                                            TestMode mode = TestMode::TEST);

/// Registers the ``StringNormalizer`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterStringNormalizerCases(std::vector<TestCase> &registry,
                                                                 TestMode mode = TestMode::TEST);

/// Registers the ``RegexFullMatch`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterRegexFullMatchCases(std::vector<TestCase> &registry,
                                                               TestMode mode = TestMode::TEST);

/// Registers the ``TfIdfVectorizer`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterTfIdfVectorizerCases(std::vector<TestCase> &registry,
                                                                TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
