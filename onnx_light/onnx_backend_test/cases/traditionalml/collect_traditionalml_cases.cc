// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectTraditionalMLTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"ArrayFeatureExtractor", &RegisterArrayFeatureExtractorCases},
      {"Binarizer", &RegisterBinarizerCases},
      {"LabelEncoder", &RegisterLabelEncoderCases},
      {"ZipMap", &RegisterZipMapCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
