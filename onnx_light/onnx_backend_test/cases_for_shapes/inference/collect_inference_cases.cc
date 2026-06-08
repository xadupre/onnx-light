// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// The shape-inference cases are multi-node graphs whose ``op_type`` cannot
// be reduced to a single representative operator. We therefore unconditionally
// register every case; the ``op_type`` filter is applied by matching the
// graph's *first* operator only, which is sufficient for the existing
// node-filtering use cases.
void CollectShapeInferenceTestCases(std::vector<TestCase> &registry, const std::string &op_type) {
  static const OpRegisterMap kEntries = {
      // First node of each multi-node graph is used as the dispatch key so
      // ``CollectTestCases("Add")``/etc. still pick up the corresponding case.
      {"shape", &RegisterAddConcatReshapeShapeInferenceCases},
      {"shape", &RegisterNonZeroChainAnonShapeInferenceCases},
      {"shape", &RegisterShapeIdentityUnsqueezeShapeInferenceCases},
      // Single node calling a model-local function. The dispatch key is
      // the function name (the op_type the caller node uses).
      {"shape", &RegisterLocalFunctionAddShapeInferenceCases},
      // Single node calling a model-local function whose body calls
      // another model-local function (nested expansion).
      {"shape", &RegisterNestedLocalFunctionAddShapeInferenceCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries);
  if (op_type.empty() or op_type == "shape") {
    RegisterNonZeroChainNamedShapeInferenceCases(registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
