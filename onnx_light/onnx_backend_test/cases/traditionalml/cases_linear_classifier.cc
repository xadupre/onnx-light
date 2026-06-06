// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_kernels/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void RegisterLinearClassifierCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::LinearClassifier cls{ctx};

  NodeProto node;
  node.set_op_type("LinearClassifier");
  node.set_domain("ai.onnx.ml");
  node.add_input("x");
  node.add_output("y");
  node.add_output("z");

  AttributeProto *coefficients = node.add_attribute();
  coefficients->set_name("coefficients");
  coefficients->set_type(AttributeProto::AttributeType::FLOATS);
  coefficients->add_floats(1.0f);
  coefficients->add_floats(-1.0f);

  AttributeProto *intercepts = node.add_attribute();
  intercepts->set_name("intercepts");
  intercepts->set_type(AttributeProto::AttributeType::FLOATS);
  intercepts->add_floats(0.0f);

  AttributeProto *multi_class = node.add_attribute();
  multi_class->set_name("multi_class");
  multi_class->set_type(AttributeProto::AttributeType::INT);
  multi_class->set_i(static_cast<int64_t>(0));

  AttributeProto *post_transform = node.add_attribute();
  post_transform->set_name("post_transform");
  post_transform->set_type(AttributeProto::AttributeType::STRING);
  post_transform->set_s("NONE");

  AttributeProto *labels = node.add_attribute();
  labels->set_name("classlabels_ints");
  labels->set_type(AttributeProto::AttributeType::INTS);
  labels->add_ints(static_cast<int64_t>(0));
  labels->add_ints(static_cast<int64_t>(1));

  Tensor x = Tensor::FromFloat("", {2, 2}, {2.0f, 1.0f, 0.0f, 3.0f});
  auto yz = cls.operator()<float>(x, {1.0f, -1.0f}, {0.0f}, std::vector<int64_t>{0, 1}, "NONE");

  Expect(node, {x}, {yz.first, yz.second}, "test_cc_linearclassifier_int64_binary",
         {default_opset, opset}, "backend-test", registry);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
