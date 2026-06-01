// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``tensor`` op category — exposed
// so individual cases live in separate translation units yet can be invoked
// from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``Concat`` backend test node case(s).
void RegisterConcatCases(std::vector<TestCase> &registry);

/// Registers the ``Cast`` backend test node case(s).
void RegisterCastCases(std::vector<TestCase> &registry);

/// Registers the ``CastLike`` backend test node case(s).
void RegisterCastLikeCases(std::vector<TestCase> &registry);

/// Registers the ``AffineGrid`` backend test node case(s).
void RegisterAffineGridCases(std::vector<TestCase> &registry);

/// Registers the ``GridSample`` backend test node case(s).
void RegisterGridSampleCases(std::vector<TestCase> &registry);

/// Registers the ``Expand`` backend test node case(s).
void RegisterExpandCases(std::vector<TestCase> &registry);

/// Registers the ``Reshape`` backend test node case(s).
void RegisterReshapeCases(std::vector<TestCase> &registry);

/// Registers the ``Slice`` backend test node case(s).
void RegisterSliceCases(std::vector<TestCase> &registry);

/// Registers the ``Tile`` backend test node case(s).
void RegisterTileCases(std::vector<TestCase> &registry);

/// Registers the ``Transpose`` backend test node case(s).
void RegisterTransposeCases(std::vector<TestCase> &registry);

/// Registers the ``Squeeze`` backend test node case(s).
void RegisterSqueezeCases(std::vector<TestCase> &registry);

/// Registers the ``Unsqueeze`` backend test node case(s).
void RegisterUnsqueezeCases(std::vector<TestCase> &registry);

/// Registers the ``NonZero`` backend test node case(s).
void RegisterNonZeroCases(std::vector<TestCase> &registry);

/// Collects all ``tensor`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectTensorTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
