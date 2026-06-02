// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectReductionTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      {"ArgMax", &RegisterArgMaxCases},
      {"ArgMin", &RegisterArgMinCases},
      {"ReduceL1", &RegisterReduceL1Cases},
      {"ReduceL2", &RegisterReduceL2Cases},
      {"ReduceMax", &RegisterReduceMaxCases},
      {"ReduceMin", &RegisterReduceMinCases},
      {"ReduceProd", &RegisterReduceProdCases},
      {"ReduceSum", &RegisterReduceSumCases},
      {"ReduceSumSquare", &RegisterReduceSumSquareCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
