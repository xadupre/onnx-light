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
      {"Col2Im", &RegisterCol2ImCases},
      {"Conv", &RegisterConvCases},
      {"ConvInteger", &RegisterConvIntegerCases},
      {"ConvTranspose", &RegisterConvTransposeCases},
      {"DeformConv", &RegisterDeformConvCases},
      {"Dropout", &RegisterDropoutCases},
      {"Flatten", &RegisterFlattenCases},
      {"GlobalAveragePool", &RegisterGlobalAveragePoolCases},
      {"GlobalLpPool", &RegisterGlobalLpPoolCases},
      {"GlobalMaxPool", &RegisterGlobalMaxPoolCases},
      {"GroupNormalization", &RegisterGroupNormalizationCases},
      {"GRU", &RegisterGRUCases},
      {"InstanceNormalization", &RegisterInstanceNormalizationCases},
      {"LRN", &RegisterLRNCases},
      {"LpNormalization", &RegisterLpNormalizationCases},
      {"LSTM", &RegisterLSTMCases},
      {"MeanVarianceNormalization", &RegisterMeanVarianceNormalizationCases},
      {"RNN", &RegisterRNNCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
