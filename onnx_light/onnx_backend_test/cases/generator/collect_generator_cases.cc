// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/generator/include_generator_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectGeneratorTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"Bernoulli", &RegisterBernoulliCases},
      {"Constant", &RegisterConstantCases},
      {"ConstantOfShape", &RegisterConstantOfShapeCases},
      {"EyeLike", &RegisterEyeLikeCases},
      {"Range", &RegisterRangeCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
