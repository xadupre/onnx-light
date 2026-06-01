// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectNNTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"Attention", &RegisterAttentionCases},
      {"AveragePool", &RegisterAveragePoolCases},
      {"BatchNormalization", &RegisterBatchNormalizationCases},
      {"Conv", &RegisterConvCases},
      {"ConvInteger", &RegisterConvIntegerCases},
      {"ConvTranspose", &RegisterConvTransposeCases},
      {"DeformConv", &RegisterDeformConvCases},
      {"Dropout", &RegisterDropoutCases},
      {"GlobalAveragePool", &RegisterGlobalAveragePoolCases},
      {"GlobalLpPool", &RegisterGlobalLpPoolCases},
      {"GlobalMaxPool", &RegisterGlobalMaxPoolCases},
      {"RNN", &RegisterRNNCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
