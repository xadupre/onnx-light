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
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterShapeTagCases(std::vector<TestCase> &registry,
                                                         TestMode mode = TestMode::TEST);

/// Registers a ``Constant → Reshape`` case whose intermediate tensor ``S``
/// should receive the ``"shape"`` value tag: the ``Constant`` node initially
/// tags it ``"weight"``, but the ``Reshape`` node consumes it as its *shape*
/// input (pushing tag ``"shape"``), and ``"shape"`` has higher priority than
/// ``"weight"``.  The ``Constant`` node itself is also tagged ``"shape"`` on
/// the second inference pass.  Graph input ``X`` and output ``Y`` are tagged
/// ``"weight"``.  The expected metadata is pre-embedded into the model so the
/// test can verify that ``WriteValueAndNodeTagsToMetadata`` produces identical
/// results.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterShapeTagAmbiguousCases(std::vector<TestCase> &registry,
                                                                  TestMode mode = TestMode::TEST);

/// Registers a ``Constant → Mul → Concat → Reshape`` case.  A ``Constant``
/// node produces the shape tensor ``S1`` (INT64 [1]), which is multiplied by
/// 2 via ``Mul`` to produce ``S2``.  ``S1`` and ``S2`` are concatenated along
/// axis 0 to form the full shape tensor ``S_full`` (INT64 [2]), which is then
/// used as the *shape* input of ``Reshape``.  Because ``S_full`` is consumed
/// as a shape input, all shape-carrying intermediates (``S1``, ``S2``,
/// ``S_full``) receive the ``"shape"`` value tag, while the scalar multiplier
/// constant ``two`` keeps its ``"weight"`` tag.  Graph input ``X`` and output
/// ``Y`` receive the ``"weight"`` tag.  The expected metadata is pre-embedded
/// into the model so the test can verify that
/// ``WriteValueAndNodeTagsToMetadata`` produces identical results.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterShapeTagConstantMulConcatReshapeCases(std::vector<TestCase> &registry,
                                              TestMode mode = TestMode::TEST);

/// Registers a case where the model output is directly a shape tensor. The
/// graph has a single ``Shape(X)`` node whose output ``Y`` is also the graph
/// output (no intermediate ``Reshape``). ``Y`` should receive the ``"shape"``
/// value tag because it is the direct output of a ``Shape`` node, exercising
/// the code path that writes ``onnx_light.value_tag`` to a graph output entry.
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterShapeTagOutputAsShapeCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers a ``Concat (weight wins)`` case. PAST (rank-3 FLOAT graph input,
/// seeded "weight") and KH (rank-3 FLOAT initializer, "weight") are concatenated
/// along axis 1.  Because KH is "weight", the output C inherits "weight"
/// (weight-wins rule).  Concat backward then also tags PAST as "weight".
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterShapeTagConcatWeightWinsCases(std::vector<TestCase> &registry,
                                      TestMode mode = TestMode::TEST);
/// Cast backward then tags X as "weight".
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterShapeTagCastBackwardCases(std::vector<TestCase> &registry, TestMode mode = TestMode::TEST);

/// Registers a ``Reshape backward`` case. X (rank-1 FLOAT graph input, seeded
/// "weight") is reshaped into Y (FLOAT [2,3]).  W (FLOAT [2,3] initializer,
/// "weight") is added to Y to produce Z.  Z inherits "weight" from W; Add
/// backward tags Y as "weight"; Reshape backward then tags X as "weight".
ONNX_LIGHT_BACKEND_TEST_LOCAL void
RegisterShapeTagReshapeBackwardCases(std::vector<TestCase> &registry,
                                     TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
