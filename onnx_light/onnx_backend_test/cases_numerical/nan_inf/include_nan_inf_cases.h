// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/test_case.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// Backend test cases that exercise operators on tensors containing the
// non-finite IEEE-754 special values (``NaN``, ``+Inf`` and ``-Inf``). These
// cases live in their own ``cases_numerical/nan_inf`` subtree so they are
// easy to discover and extend with other numerically-oriented scenarios
// later (the parent ``cases_numerical`` directory is reserved for other
// special-value families such as denormals or signed zero).
//
// For each element-wise binary math operator we register a small case that
// pairs the special values with regular finite operands so the
// kernel/reference comparison exercises both well-defined results
// (e.g. ``1 + Inf == Inf``) and propagation of ``NaN``. ``Where`` is also
// covered because the issue explicitly asks for it: the condition tensor
// selects between ``x`` and ``y`` tensors containing ``NaN``/``+Inf``/
// ``-Inf``, ensuring the operator forwards the special values unchanged.
//
// Expected outputs are computed by the in-tree kernel so each case stays
// self-consistent with the implementation under test.
// ---------------------------------------------------------------------------

/// Registers backend test cases that ``Add`` tensors containing NaN/Inf.
void RegisterAddNanInfCases(std::vector<TestCase> &registry);

/// Registers backend test cases that ``Sub`` tensors containing NaN/Inf.
void RegisterSubNanInfCases(std::vector<TestCase> &registry);

/// Registers backend test cases that ``Mul`` tensors containing NaN/Inf.
void RegisterMulNanInfCases(std::vector<TestCase> &registry);

/// Registers backend test cases that ``Div`` tensors containing NaN/Inf.
void RegisterDivNanInfCases(std::vector<TestCase> &registry);

/// Registers backend test cases that run the ``Where`` node on tensors
/// containing NaN/Inf in either branch.
void RegisterWhereNanInfCases(std::vector<TestCase> &registry);

/// Collects all NaN/Inf backend test cases by invoking every
/// ``Register*NanInfCases`` helper declared in this header.
void CollectNanInfTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
