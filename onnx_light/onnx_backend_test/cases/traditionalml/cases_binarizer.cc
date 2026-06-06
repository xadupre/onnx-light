// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Binarizer — y[i] = 1 if x[i] > threshold else 0 (element-wise; same dtype
// and shape as the input). Mirrors the upstream ONNX node test
// ``test_ai_onnx_ml_binarizer`` (see
// ``onnx/backend/test/case/node/ai_onnx_ml/binarizer.py``), which uses
// threshold=1.0 on a ``(3, 4, 5)`` ``float32`` input.
// ---------------------------------------------------------------------------
void RegisterBinarizerCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::Binarizer binarizer{ctx};

  // Canonical float case mirroring the upstream ONNX node test fixture.
  {
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

    // Small representative slice of the upstream (3, 4, 5) input. Using a
    // compact ``(2, 4)`` tensor keeps the test case self-contained while
    // still exercising both ``x[i] > threshold`` and ``x[i] <= threshold``
    // branches of the kernel.
    Tensor x = Tensor::FromFloat("", {2, 4}, {0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 1.0f, 3.0f, 0.9f});
    Tensor y = binarizer.operator()<float>(x, threshold);

    Expect(node, {x}, {y}, "test_cc_binarizer_float", {default_opset, opset}, "backend-test",
           registry);
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
    threshold_attr->set_name("threshold");
    threshold_attr->set_type(AttributeProto::AttributeType::FLOAT);
    threshold_attr->set_f(static_cast<float>(threshold));

    Tensor x = Tensor::FromInt64("", {5}, {0, 3, 4, -2, 10});
    Tensor y = binarizer.operator()<int64_t>(x, threshold);

    Expect(node, {x}, {y}, "test_cc_binarizer_int64", {default_opset, opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
