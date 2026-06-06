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

void RegisterSVMClassifierCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::SVMClassifier svm{ctx};

  NodeProto node;
  node.set_op_type("SVMClassifier");
  node.set_domain("ai.onnx.ml");
  node.add_input("x");
  node.add_output("y");
  node.add_output("z");

  AttributeProto *kernel_type = node.add_attribute();
  kernel_type->set_name("kernel_type");
  kernel_type->set_type(AttributeProto::AttributeType::STRING);
  kernel_type->set_s("LINEAR");

  AttributeProto *kernel_params = node.add_attribute();
  kernel_params->set_name("kernel_params");
  kernel_params->set_type(AttributeProto::AttributeType::FLOATS);
  kernel_params->add_floats(0.0f);
  kernel_params->add_floats(0.0f);
  kernel_params->add_floats(0.0f);

  AttributeProto *vectors_per_class = node.add_attribute();
  vectors_per_class->set_name("vectors_per_class");
  vectors_per_class->set_type(AttributeProto::AttributeType::INTS);
  vectors_per_class->add_ints(static_cast<int64_t>(1));
  vectors_per_class->add_ints(static_cast<int64_t>(1));

  AttributeProto *support_vectors = node.add_attribute();
  support_vectors->set_name("support_vectors");
  support_vectors->set_type(AttributeProto::AttributeType::FLOATS);
  support_vectors->add_floats(1.0f);
  support_vectors->add_floats(0.0f);
  support_vectors->add_floats(0.0f);
  support_vectors->add_floats(1.0f);

  AttributeProto *coefficients = node.add_attribute();
  coefficients->set_name("coefficients");
  coefficients->set_type(AttributeProto::AttributeType::FLOATS);
  coefficients->add_floats(1.0f);
  coefficients->add_floats(-1.0f);

  AttributeProto *rho = node.add_attribute();
  rho->set_name("rho");
  rho->set_type(AttributeProto::AttributeType::FLOATS);
  rho->add_floats(0.0f);

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
  auto yz = svm.operator()<float>(x, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, -1.0f}, {0.0f}, {1, 1},
                                  {0, 1}, "LINEAR", 0.0f, 0.0f, 0.0f);

  Expect(node, {x}, {yz.first, yz.second}, "test_cc_svmclassifier_int64_binary",
         {default_opset, opset}, "backend-test", registry);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
