// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/training/include_training_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectTrainingTestCases(std::vector<TestCase> &registry) { RegisterAdamCases(registry); }

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
