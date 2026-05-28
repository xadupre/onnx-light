// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/controlflow/include_controlflow_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void CollectControlflowTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  DispatchRegisterByOpType(registry, op_type,
                           {
                               {"If", &RegisterIfCases},
                           });
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
