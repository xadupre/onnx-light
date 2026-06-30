// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Backend test cases that exercise the shape-tag annotation metadata stored
// in ``GraphProto::metadata_props`` (``onnx_light.value_tags``) and in
// ``NodeProto::metadata_props`` (``onnx_light.node_tag``). These live in
// their own ``cases_for_shapes`` subtree so callers can collect them
// independently from the broader shape-inference gallery.
// ---------------------------------------------------------------------------

/// Registers a ``Shape → Reshape`` case whose intermediate tensor ``S``
/// should receive the ``"shape"`` value tag (because it is the output of a
/// ``Shape`` node) and whose ``Shape`` node should receive the
/// ``"shape"`` node tag. The expected tag metadata is pre-embedded into the
/// model so the test can verify that
/// ``WriteValueAndNodeTagsToMetadata`` produces identical results.
void RegisterShapeTagCases(std::vector<TestCase> &registry);

/// Registers a ``Shape → Reshape`` case that exercises the
/// ``kReleaseShapeTag`` event path. The shape tensor ``S`` is released at
/// the ``Reshape`` node (its last consumer) and is tagged ``"shape"``, so
/// ``ComputeInPlaceReuseGraph`` must emit a ``kReleaseShapeTag`` event for
/// it. The expected ``onnx_light.release_after`` and
/// ``onnx_light.release_after_shape_tag`` node metadata are pre-embedded so
/// tests can verify that ``ComputeContext::WriteToMetadata`` reproduces them.
void RegisterShapeTagReleaseEventCases(std::vector<TestCase> &registry);

/// Collects all shape-tag backend test cases by invoking every
/// ``Register*ShapeTag*Cases`` helper declared in this header.
void CollectShapeTagTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
