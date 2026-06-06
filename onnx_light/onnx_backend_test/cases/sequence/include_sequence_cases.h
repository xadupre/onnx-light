// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``sequence`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``SequenceConstruct`` backend test node case(s).
void RegisterSequenceConstructCases(std::vector<TestCase> &registry);

/// Registers the ``SequenceEmpty`` backend test node case(s).
void RegisterSequenceEmptyCases(std::vector<TestCase> &registry);

/// Registers the ``ConcatFromSequence`` backend test node case(s).
void RegisterConcatFromSequenceCases(std::vector<TestCase> &registry);

/// Registers the ``SequenceLength`` backend test node case(s).
void RegisterSequenceLengthCases(std::vector<TestCase> &registry);

/// Registers the ``SequenceErase`` backend test node case(s).
void RegisterSequenceEraseCases(std::vector<TestCase> &registry);

/// Registers the ``SequenceAt`` backend test node case(s).
void RegisterSequenceAtCases(std::vector<TestCase> &registry);

/// Registers the ``SequenceInsert`` backend test node case(s).
void RegisterSequenceInsertCases(std::vector<TestCase> &registry);

/// Registers the ``SequenceMap`` backend test node case(s).
void RegisterSequenceMapCases(std::vector<TestCase> &registry);

/// Registers the ``SplitToSequence`` backend test node case(s).
void RegisterSplitToSequenceCases(std::vector<TestCase> &registry);

/// Collects all ``sequence`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectSequenceTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
