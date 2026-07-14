// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``generator`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Bernoulli`` backend test node case(s).
void RegisterBernoulliCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Constant`` backend test node case(s).
void RegisterConstantCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``ConstantOfShape`` backend test node case(s).
void RegisterConstantOfShapeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``EyeLike`` backend test node case(s).
void RegisterEyeLikeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``RandomNormal`` backend test node case(s).
void RegisterRandomNormalCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``RandomNormalLike`` backend test node case(s).
void RegisterRandomNormalLikeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``RandomUniform`` backend test node case(s).
void RegisterRandomUniformCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``RandomUniformLike`` backend test node case(s).
void RegisterRandomUniformLikeCases(std::vector<TestCase> &registry,
                                    TestMode mode = TestMode::TEST);

/// Registers the ``Range`` backend test node case(s).
void RegisterRangeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Multinomial`` backend test node case(s).
void RegisterMultinomialCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``ai.rt::DelayedInitializer`` backend test node case(s).
void RegisterDelayedInitializerCases(std::vector<TestCase> &registry,
                                     TestMode mode = TestMode::TEST);

/// Collects all ``generator`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectGeneratorTestCases(std::vector<TestCase> &registry, const std::string &op_type = "",
                               TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
