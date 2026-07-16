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
// Per-operator registration helpers for the ``preview`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``ai.onnx.preview::FlexAttention`` backend test node case(s).
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterFlexAttentionCases(std::vector<TestCase> &registry,
                                                              TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
