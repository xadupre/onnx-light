// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``quantization`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``QuantizeLinear`` backend test node case(s).
void RegisterQuantizeLinearCases(std::vector<TestCase> &registry);

/// Registers the ``DequantizeLinear`` backend test node case(s).
void RegisterDequantizeLinearCases(std::vector<TestCase> &registry);

/// Registers the ``DynamicQuantizeLinear`` backend test node case(s).
void RegisterDynamicQuantizeLinearCases(std::vector<TestCase> &registry);

/// Registers the ``QLinearMatMul`` backend test node case(s).
void RegisterQLinearMatMulCases(std::vector<TestCase> &registry);

/// Registers the ``QLinearConv`` backend test node case(s).
void RegisterQLinearConvCases(std::vector<TestCase> &registry);

/// Collects all ``quantization`` op category backend test node cases by
/// invoking every ``Register*Cases`` helper declared in this header.
void CollectQuantizationTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
