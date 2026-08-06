// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/text/include_text_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectTextTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                          TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"StringConcat", &RegisterStringConcatCases},
      {"StringSplit", &RegisterStringSplitCases},
      {"StringNormalizer", &RegisterStringNormalizerCases},
      {"RegexFullMatch", &RegisterRegexFullMatchCases},
      {"TfIdfVectorizer", &RegisterTfIdfVectorizerCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
