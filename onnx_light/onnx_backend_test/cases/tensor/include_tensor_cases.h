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

/// Registers the ``BitCast`` backend test node case(s).
void RegisterBitCastCases(std::vector<TestCase> &registry);

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

/// Registers the ``Upsample`` backend test node case(s).
void RegisterUpsampleCases(std::vector<TestCase> &registry);

/// Registers the ``Transpose`` backend test node case(s).
void RegisterTransposeCases(std::vector<TestCase> &registry);

/// Registers the ``Trilu`` backend test node case(s).
void RegisterTriluCases(std::vector<TestCase> &registry);

/// Registers the ``ReverseSequence`` backend test node case(s).
void RegisterReverseSequenceCases(std::vector<TestCase> &registry);

/// Registers the ``DepthToSpace`` backend test node case(s).
void RegisterDepthToSpaceCases(std::vector<TestCase> &registry);

/// Registers the ``SpaceToDepth`` backend test node case(s).
void RegisterSpaceToDepthCases(std::vector<TestCase> &registry);

/// Registers the ``Squeeze`` backend test node case(s).
void RegisterSqueezeCases(std::vector<TestCase> &registry);

/// Registers the ``Unsqueeze`` backend test node case(s).
void RegisterUnsqueezeCases(std::vector<TestCase> &registry);

/// Registers the ``NonZero`` backend test node case(s).
void RegisterNonZeroCases(std::vector<TestCase> &registry);

/// Registers the ``Unique`` backend test node case(s).
void RegisterUniqueCases(std::vector<TestCase> &registry);

/// Registers the ``Shape`` backend test node case(s).
void RegisterShapeCases(std::vector<TestCase> &registry);

/// Registers the ``Gather`` backend test node case(s).
void RegisterGatherCases(std::vector<TestCase> &registry);

/// Registers the ``GatherElements`` backend test node case(s).
void RegisterGatherElementsCases(std::vector<TestCase> &registry);

/// Registers the ``GatherND`` backend test node case(s).
void RegisterGatherNDCases(std::vector<TestCase> &registry);

/// Registers the ``Compress`` backend test node case(s).
void RegisterCompressCases(std::vector<TestCase> &registry);

/// Registers the ``Split`` backend test node case(s).
void RegisterSplitCases(std::vector<TestCase> &registry);

/// Registers the ``TensorScatter`` backend test node case(s).
void RegisterTensorScatterCases(std::vector<TestCase> &registry);

/// Registers the ``ScatterElements`` backend test node case(s).
void RegisterScatterElementsCases(std::vector<TestCase> &registry);

/// Registers the ``ScatterND`` backend test node case(s).
void RegisterScatterNDCases(std::vector<TestCase> &registry);

/// Collects all ``tensor`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectTensorTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
