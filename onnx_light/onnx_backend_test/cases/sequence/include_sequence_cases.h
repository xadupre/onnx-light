// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``sequence`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``SequenceConstruct`` backend test node case(s).
void RegisterSequenceConstructCases(std::vector<TestCase> &registry);

/// Registers the ``ConcatFromSequence`` backend test node case(s).
void RegisterConcatFromSequenceCases(std::vector<TestCase> &registry);

/// Collects all ``sequence`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectSequenceTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
