// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/generator/include_generator_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectGeneratorTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                               TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"Bernoulli", &RegisterBernoulliCases},
      {"Constant", &RegisterConstantCases},
      {"ConstantOfShape", &RegisterConstantOfShapeCases},
      {"DelayedInitializer", &RegisterDelayedInitializerCases},
      {"EyeLike", &RegisterEyeLikeCases},
      {"RandomNormal", &RegisterRandomNormalCases},
      {"RandomNormalLike", &RegisterRandomNormalLikeCases},
      {"RandomUniform", &RegisterRandomUniformCases},
      {"RandomUniformLike", &RegisterRandomUniformLikeCases},
      {"Range", &RegisterRangeCases},
      {"Multinomial", &RegisterMultinomialCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
