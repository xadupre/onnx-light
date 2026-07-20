// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Compatibility shim: TestCase and its helpers are now defined in
// ``onnx_core/backend_test/test_case.h`` under the ``core::backend_test``
// namespace.  This header re-exports everything into the legacy
// ``onnx_backend_test`` namespace so existing case files in
// ``onnx_backend_test/cases/`` continue to compile unchanged.

#pragma once

#include "onnx_core/backend_test/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// The blanket ``using namespace`` below is intentional and required:
// case files open ``namespace onnx_backend_test { ... }`` blocks and use
// names such as ``Tensor`` or ``TypeSpec`` without qualification.  Those
// names live transitively in ``core::runtime`` / ``core::backend_test``
// and are only reachable via the unqualified-lookup chain that a
// namespace-scope ``using namespace`` establishes.
//
// The explicit using-declarations that follow are additionally needed for
// qualified access (``onnx_backend_test::TestCase`` etc.) from callers
// that do not open the namespace, because a ``using namespace`` directive
// does not affect qualified name lookup.
using namespace ::onnx_light::core::backend_test; // NOLINT(google-build-using-namespace)

using DataSet = ::onnx_light::core::backend_test::DataSet;
using OpsetId = ::onnx_light::core::backend_test::OpsetId;
using TestCase = ::onnx_light::core::backend_test::TestCase;
using TestMode = ::onnx_light::core::backend_test::TestMode;
using ::onnx_light::core::backend_test::CollectTestCases;
using ::onnx_light::core::backend_test::CollectTestCasesByName;
using ::onnx_light::core::backend_test::DefaultOpset;
using ::onnx_light::core::backend_test::Expect;

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
