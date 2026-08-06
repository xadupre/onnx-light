// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/logical/include_logical_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectLogicalTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                             TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"And", &RegisterAndCases},
      {"Or", &RegisterOrCases},
      {"Xor", &RegisterXorCases},
      {"Greater", &RegisterGreaterCases},
      {"GreaterOrEqual", &RegisterGreaterOrEqualCases},
      {"Less", &RegisterLessCases},
      {"LessOrEqual", &RegisterLessOrEqualCases},
      {"Equal", &RegisterEqualCases},
      {"Where", &RegisterWhereCases},
      {"Not", &RegisterNotCases},
      {"IsNaN", &RegisterIsNaNCases},
      {"IsInf", &RegisterIsInfCases},
      {"BitwiseAnd", &RegisterBitwiseAndCases},
      {"BitwiseOr", &RegisterBitwiseOrCases},
      {"BitwiseXor", &RegisterBitwiseXorCases},
      {"BitwiseNot", &RegisterBitwiseNotCases},
      {"BitShift", &RegisterBitShiftCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
