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
void RegisterConcatCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Cast`` backend test node case(s).
void RegisterCastCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``CastLike`` backend test node case(s).
void RegisterCastLikeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``BitCast`` backend test node case(s).
void RegisterBitCastCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``AffineGrid`` backend test node case(s).
void RegisterAffineGridCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``GridSample`` backend test node case(s).
void RegisterGridSampleCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Expand`` backend test node case(s).
void RegisterExpandCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Reshape`` backend test node case(s).
void RegisterReshapeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Slice`` backend test node case(s).
void RegisterSliceCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Tile`` backend test node case(s).
void RegisterTileCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Pad`` backend test node case(s).
void RegisterPadCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Upsample`` backend test node case(s).
void RegisterUpsampleCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Resize`` backend test node case(s).
void RegisterResizeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Transpose`` backend test node case(s).
void RegisterTransposeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Trilu`` backend test node case(s).
void RegisterTriluCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``CenterCropPad`` backend test node case(s).
void RegisterCenterCropPadCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``ReverseSequence`` backend test node case(s).
void RegisterReverseSequenceCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``DepthToSpace`` backend test node case(s).
void RegisterDepthToSpaceCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``SpaceToDepth`` backend test node case(s).
void RegisterSpaceToDepthCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Squeeze`` backend test node case(s).
void RegisterSqueezeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Unsqueeze`` backend test node case(s).
void RegisterUnsqueezeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``NonZero`` backend test node case(s).
void RegisterNonZeroCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``OneHot`` backend test node case(s).
void RegisterOneHotCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Unique`` backend test node case(s).
void RegisterUniqueCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Shape`` backend test node case(s).
void RegisterShapeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Size`` backend test node case(s).
void RegisterSizeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Identity`` backend test node case(s).
void RegisterIdentityCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Gather`` backend test node case(s).
void RegisterGatherCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``GatherElements`` backend test node case(s).
void RegisterGatherElementsCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``GatherND`` backend test node case(s).
void RegisterGatherNDCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Compress`` backend test node case(s).
void RegisterCompressCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``Split`` backend test node case(s).
void RegisterSplitCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``TensorScatter`` backend test node case(s).
void RegisterTensorScatterCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``ScatterElements`` backend test node case(s).
void RegisterScatterElementsCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the deprecated ``Scatter`` (opset 9, deprecated since opset 11)
/// backend test node case(s).
void RegisterScatterCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers the ``ScatterND`` backend test node case(s).
void RegisterScatterNDCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Collects all ``tensor`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
ONNX_LIGHT_BACKEND_TEST_API void CollectTensorTestCases(std::vector<TestCase> &registry,
                                                        const std::string &op_type = "",
                                                        TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
