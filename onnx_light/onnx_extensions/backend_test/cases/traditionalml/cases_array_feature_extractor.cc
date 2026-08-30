// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterArrayFeatureExtractorCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);
  const auto afe = MakeReferenceKernel<onnx_kernels::kernel::ArrayFeatureExtractor>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("ArrayFeatureExtractor");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Expect(registry, std::move(node), "test_ai_onnx_ml_array_feature_extractor_benchmark",
           {default_opset, opset}, {32768, 3}, {24576}, [afe]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {8192, 4}, 2701);
             Tensor y = Tensor::FromInt64("", {3}, {0, 2, 3});
             Tensor z = afe.Invoke(
                 [&](const auto &kernel) { return kernel.template operator()<float>(x, y); });
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  NodeProto node;
  node.set_op_type("ArrayFeatureExtractor");
  node.set_domain("ai.onnx.ml");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");

  Expect(registry, std::move(node), "test_ai_onnx_ml_array_feature_extractor",
         {default_opset, opset}, [=]() -> IoData {
           Tensor x = Tensor::FromFloat(
               "", {3, 4},
               {0.0f, 1.0f, 2.0f, 3.0f, 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f});
           Tensor y = Tensor::FromInt64("", {3}, {0, 2, 3});
           Tensor z = afe.Invoke(
               [&](const auto &kernel) { return kernel.template operator()<float>(x, y); });
           return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
         });
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
