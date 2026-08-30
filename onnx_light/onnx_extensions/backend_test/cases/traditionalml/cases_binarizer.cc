// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// Binarizer — y[i] = 1 if x[i] > threshold else 0 (element-wise; same dtype
// and shape as the input). Mirrors the upstream ONNX node test
// ``test_ai_onnx_ml_binarizer`` (see
// ``onnx/backend/test/case/node/ai_onnx_ml/binarizer.py``), which uses
// threshold=1.0 on a ``(3, 4, 5)`` ``float32`` input.
// ---------------------------------------------------------------------------
void RegisterBinarizerCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);
  const auto binarizer = MakeReferenceKernel<onnx_kernels::kernel::Binarizer>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Binarizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const float threshold = 1.0f;
    AttributeProto *threshold_attr = node.add_attribute();
    threshold_attr->set_name("threshold");
    threshold_attr->set_type(AttributeProto::AttributeType::FLOAT);
    threshold_attr->set_f(threshold);

    Expect(registry, std::move(node), "test_ai_onnx_ml_binarizer_benchmark", {default_opset, opset},
           {32768}, {32768}, [binarizer, threshold]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {8192, 4}, 2611);
             Tensor y = binarizer.Invoke([&](const auto &kernel) {
               return kernel.template operator()<float>(x, threshold);
             });
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Canonical float case mirroring the upstream ONNX node test fixture.
  {
    NodeProto node;
    node.set_op_type("Binarizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const float threshold = 1.0f;
    AttributeProto *threshold_attr = node.add_attribute();
    Expect(registry, std::move(node), "test_ai_onnx_ml_binarizer", {default_opset, opset},
           [=]() -> IoData {
             threshold_attr->set_name("threshold");
             threshold_attr->set_type(AttributeProto::AttributeType::FLOAT);
             threshold_attr->set_f(threshold);

             // Small representative slice of the upstream (3, 4, 5) input. Using a
             // compact ``(2, 4)`` tensor keeps the test case self-contained while
             // still exercising both ``x[i] > threshold`` and ``x[i] <= threshold``
             // branches of the kernel.
             Tensor x =
                 Tensor::FromFloat("", {2, 4}, {0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 1.0f, 3.0f, 0.9f});
             Tensor y = binarizer.Invoke([&](const auto &kernel) {
               return kernel.template operator()<float>(x, threshold);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // Int64 case: exercises the non-default element type and the equality edge
  // (``x[i] == threshold`` must map to 0, not 1).
  {
    NodeProto node;
    node.set_op_type("Binarizer");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const int64_t threshold = 3;
    AttributeProto *threshold_attr = node.add_attribute();
    Expect(registry, std::move(node), "test_cc_binarizer_int64", {default_opset, opset},
           [=]() -> IoData {
             threshold_attr->set_name("threshold");
             threshold_attr->set_type(AttributeProto::AttributeType::FLOAT);
             threshold_attr->set_f(static_cast<float>(threshold));

             Tensor x = Tensor::FromInt64("", {5}, {0, 3, 4, -2, 10});
             Tensor y = binarizer.Invoke([&](const auto &kernel) {
               return kernel.template operator()<int64_t>(x, threshold);
             });

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
