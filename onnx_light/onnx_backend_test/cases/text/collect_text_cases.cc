// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/text/include_text_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void CollectTextTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"StringConcat", &RegisterStringConcatCases},
      {"StringSplit", &RegisterStringSplitCases},
      {"StringNormalizer", &RegisterStringNormalizerCases},
      {"RegexFullMatch", &RegisterRegexFullMatchCases},
      {"TfIdfVectorizer", &RegisterTfIdfVectorizerCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
