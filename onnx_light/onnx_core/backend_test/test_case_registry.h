// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/backend_test/test_case.h"

#include <functional>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

/**
 * Collector function type used by the registration mechanism.
 *
 * A collector receives the output registry, an optional operator-type filter,
 * the ``include_big`` flag and the :ref:`TestMode`, and appends the matching
 * :ref:`TestCase` entries to ``registry``. Registered via
 * :func:`RegisterTestCasesCollector`; invoked by :func:`CollectTestCases`.
 */
using TestCasesCollectorFn =
    std::function<void(std::vector<TestCase> &, const std::string &, bool, TestMode)>;

/**
 * Registers a collector function into the global per-category registry.
 * Called once per category at static-initialization time (typically via a
 * ``static int kRegXxx = RegisterTestCasesCollector(...)`` variable in each
 * ``collect_*_cases.cc`` translation unit).
 *
 * @return 0; the return value exists solely to allow the static-initializer
 *         idiom.
 */
int RegisterTestCasesCollector(TestCasesCollectorFn fn);

/**
 * Returns a read-only view of the registered collector functions.
 *
 * Called by :func:`CollectTestCases` (defined in
 * ``onnx_backend_test/collect_test_cases.cc``) to iterate the functions that
 * were registered via :func:`RegisterTestCasesCollector`. Exposed here so that
 * ``collect_test_cases.cc`` (compiled into ``lib_onnx_backend_test``) can
 * access the registry without a circular dependency.
 */
const std::vector<TestCasesCollectorFn> &GetRegisteredCollectors();

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test
