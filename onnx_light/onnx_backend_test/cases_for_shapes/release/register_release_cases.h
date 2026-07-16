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
// Backend test cases that exercise the release-after metadata written by
// ``ComputeContext::ComputeInPlaceReuseGraph`` when an intermediate tensor
// reaches its last consumer.  These live in their own ``cases_for_shapes``
// subtree so callers can collect them independently from the broader
// shape-inference gallery.
// ---------------------------------------------------------------------------

/// Registers a ``Shape → Reshape`` case whose intermediate tensor ``S``
/// is released at the ``Reshape`` node (its only consumer).  The expected
/// ``onnx_light.release_after`` and ``onnx_light.not_used_after`` node metadata
/// are pre-embedded in the model so tests can verify that
/// ``ComputeContext::ComputeInPlaceReuseGraph`` reproduces them.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterReleaseCases(std::vector<TestCase> &registry,
                                                        TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
