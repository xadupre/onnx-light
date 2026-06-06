// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/reduction/include_reduction_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void CollectReductionTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  // clang-format off
  static const OpRegisterMap kEntries = {
      {"ArgMax", &RegisterArgMaxCases},
      {"ArgMin", &RegisterArgMinCases},
      {"ReduceL1", &RegisterReduceL1Cases},
      {"ReduceL2", &RegisterReduceL2Cases},
      {"ReduceLogSum", &RegisterReduceLogSumCases},
      {"ReduceLogSumExp", &RegisterReduceLogSumExpCases},
      {"ReduceMax", &RegisterReduceMaxCases},
      {"ReduceMean", &RegisterReduceMeanCases},
      {"ReduceMin", &RegisterReduceMinCases},
      {"ReduceProd", &RegisterReduceProdCases},
      {"ReduceSum", &RegisterReduceSumCases},
      {"ReduceSumSquare", &RegisterReduceSumSquareCases},
  };
  // clang-format on
  DispatchRegisterByOpType(registry, op_type, kEntries);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
