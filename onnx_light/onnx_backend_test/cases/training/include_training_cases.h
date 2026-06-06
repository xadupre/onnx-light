// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// Per-operator registration helpers for the ``training`` op category —
// exposed so individual cases live in separate translation units yet can be
// invoked from ``CollectTestCases()``.
// ---------------------------------------------------------------------------

/// Registers the ``ai.onnx.preview.training::Adam`` backend test node case(s).
void RegisterAdamCases(std::vector<TestCase> &registry);

/// Registers the ``ai.onnx.preview.training::Adagrad`` backend test node case(s).
void RegisterAdagradCases(std::vector<TestCase> &registry);

/// Registers the ``ai.onnx.preview.training::Momentum`` backend test node case(s).
void RegisterMomentumCases(std::vector<TestCase> &registry);

/// Collects all ``training`` op category backend test node cases by invoking
/// every ``Register*Cases`` helper declared in this header.
void CollectTrainingTestCases(std::vector<TestCase> &registry, const std::string &op_type = "");

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
