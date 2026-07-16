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
// Backend test cases that exercise the in-place-reuse analysis metadata stored
// in ``NodeProto::metadata_props``. These live in their own
// ``cases_for_shapes`` subtree so callers can collect them independently from
// the broader shape-inference gallery.
// ---------------------------------------------------------------------------

/// Registers an ``Abs → Abs → Abs`` case whose intermediate tensors all share
/// the same shape so in-place-reuse inference can detect the recyclable
/// buffers and record the expected metadata on the graph nodes.
ONNX_LIGHT_BACKEND_TEST_LOCAL void RegisterInPlaceReuseCases(std::vector<TestCase> &registry,
                                                             TestMode mode = TestMode::TEST);

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
