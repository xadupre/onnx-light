// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``nn`` (neural network) op
// category — exposed so individual cases live in separate translation units
// yet can be invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``AveragePool`` backend test node case(s).
void RegisterAveragePoolCases(std::vector<TestCase> &registry);

/// Registers the ``Attention`` backend test node case(s).
void RegisterAttentionCases(std::vector<TestCase> &registry);

/// Registers the ``BatchNormalization`` backend test node case(s).
void RegisterBatchNormalizationCases(std::vector<TestCase> &registry);

/// Registers the ``DeformConv`` backend test node case(s).
void RegisterDeformConvCases(std::vector<TestCase> &registry);

/// Registers the ``Conv`` backend test node case(s).
void RegisterConvCases(std::vector<TestCase> &registry);

/// Registers the ``ConvInteger`` backend test node case(s).
void RegisterConvIntegerCases(std::vector<TestCase> &registry);

/// Registers the ``ConvTranspose`` backend test node case(s).
void RegisterConvTransposeCases(std::vector<TestCase> &registry);

/// Registers the ``Dropout`` backend test node case(s).
void RegisterDropoutCases(std::vector<TestCase> &registry);

/// Registers the ``GlobalAveragePool`` backend test node case(s).
void RegisterGlobalAveragePoolCases(std::vector<TestCase> &registry);

/// Registers the ``GlobalLpPool`` backend test node case(s).
void RegisterGlobalLpPoolCases(std::vector<TestCase> &registry);

/// Registers the ``GlobalMaxPool`` backend test node case(s).
void RegisterGlobalMaxPoolCases(std::vector<TestCase> &registry);

/// Registers the ``RNN`` backend test node case(s).
void RegisterRNNCases(std::vector<TestCase> &registry);

/// Collects all ``nn`` op category backend test node cases by invoking every
/// ``Register*Cases`` helper declared in this header.
void CollectNNTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
