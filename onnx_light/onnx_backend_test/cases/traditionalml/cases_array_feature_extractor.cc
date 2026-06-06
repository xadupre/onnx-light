// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterArrayFeatureExtractorCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::ArrayFeatureExtractor afe{ctx};

  NodeProto node;
  node.set_op_type("ArrayFeatureExtractor");
  node.set_domain("ai.onnx.ml");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  Tensor x = Tensor::FromFloat(
      "", {3, 4}, {0.0f, 1.0f, 2.0f, 3.0f, 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f});
  Tensor y = Tensor::FromInt64("", {3}, {0, 2, 3});
  Tensor z = afe.operator()<float>(x, y);

  Expect(node, {x, y}, {z}, "test_ai_onnx_ml_array_feature_extractor", {default_opset, opset},
         "backend-test", registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
