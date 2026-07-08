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

/// Registers a ``Constant → Reshape`` case whose intermediate tensor ``S``
/// should receive the ``"ambiguous"`` value tag because the ``Constant``
/// node's output is tagged ``"weight"`` while the ``Reshape`` node consumes
/// it as its *shape* input (pushing tag ``"shape"``); the conflict promotes
/// it to ``"ambiguous"``. The ``Constant`` node itself is also tagged
/// ``"ambiguous"`` on the second inference pass. The expected metadata is
/// pre-embedded into the model so the test can verify that
/// ``WriteValueAndNodeTagsToMetadata`` produces identical results.
void RegisterShapeTagAmbiguousCases(std::vector<TestCase> &registry);

/// Registers a ``Constant → Mul → Concat → Reshape`` case.  A ``Constant``
/// node produces the shape tensor ``S1`` (INT64 [1]), which is multiplied by
/// 2 via ``Mul`` to produce ``S2``.  ``S1`` and ``S2`` are concatenated along
/// axis 0 to form the full shape tensor ``S_full`` (INT64 [2]), which is then
/// used as the *shape* input of ``Reshape``.  Because ``S_full`` is consumed
/// as a shape input, all shape-carrying intermediates (``S1``, ``S2``,
/// ``S_full``) receive the ``"shape"`` value tag, while the scalar multiplier
/// constant ``two`` keeps its ``"weight"`` tag.  The expected metadata is
/// pre-embedded into the model so the test can verify that
/// ``WriteValueAndNodeTagsToMetadata`` produces identical results.
void RegisterShapeTagConstantMulConcatReshapeCases(std::vector<TestCase> &registry);

/// Registers a case where the model output is directly a shape tensor. The
/// graph has a single ``Shape(X)`` node whose output ``Y`` is also the graph
/// output (no intermediate ``Reshape``). ``Y`` should receive the ``"shape"``
/// value tag because it is the direct output of a ``Shape`` node, exercising
/// the code path that writes ``onnx_light.value_tag`` to a graph output entry.
void RegisterShapeTagOutputAsShapeCases(std::vector<TestCase> &registry);

/// Collects all shape-tag backend test cases by invoking every
/// ``Register*ShapeTag*Cases`` helper declared in this header.
void CollectShapeTagTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
