// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``reduction`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``ArgMax`` backend test node case(s).
void RegisterArgMaxCases(std::vector<TestCase> &registry);

/// Registers the ``ArgMin`` backend test node case(s).
void RegisterArgMinCases(std::vector<TestCase> &registry);

/// Registers the ``ReduceSum`` backend test node case(s).
void RegisterReduceSumCases(std::vector<TestCase> &registry);

/// Registers the ``ReduceMax`` backend test node case(s).
void RegisterReduceMaxCases(std::vector<TestCase> &registry);

/// Registers the ``ReduceMean`` backend test node case(s).
void RegisterReduceMeanCases(std::vector<TestCase> &registry);

/// Registers the ``ReduceMin`` backend test node case(s).
void RegisterReduceMinCases(std::vector<TestCase> &registry);

/// Registers the ``ReduceProd`` backend test node case(s).
void RegisterReduceProdCases(std::vector<TestCase> &registry);

/// Registers the ``ReduceL1`` backend test node case(s).
void RegisterReduceL1Cases(std::vector<TestCase> &registry);

/// Registers the ``ReduceL2`` backend test node case(s).
void RegisterReduceL2Cases(std::vector<TestCase> &registry);

/// Registers the ``ReduceLogSum`` backend test node case(s).
void RegisterReduceLogSumCases(std::vector<TestCase> &registry);

/// Registers the ``ReduceLogSumExp`` backend test node case(s).
void RegisterReduceLogSumExpCases(std::vector<TestCase> &registry);

/// Registers the ``ReduceSumSquare`` backend test node case(s).
void RegisterReduceSumSquareCases(std::vector<TestCase> &registry);

/// Collects all ``reduction`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectReductionTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
