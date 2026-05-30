// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterSVMRegressorCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::SVMRegressor svm{ctx};

  NodeProto node;
  node.set_op_type("SVMRegressor");
  node.set_domain("ai.onnx.ml");
  node.add_input("x");
  node.add_output("y");

  AttributeProto *kernel_type = node.add_attribute();
  kernel_type->set_name("kernel_type");
  kernel_type->set_type(AttributeProto::AttributeType::STRING);
  kernel_type->set_s("LINEAR");

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
  coefficients->add_floats(2.0f);
  coefficients->add_floats(-1.0f);

  AttributeProto *rho = node.add_attribute();
  rho->set_name("rho");
  rho->set_type(AttributeProto::AttributeType::FLOATS);
  rho->add_floats(0.5f);

  AttributeProto *n_supports = node.add_attribute();
  n_supports->set_name("n_supports");
  n_supports->set_type(AttributeProto::AttributeType::INT);
  n_supports->set_i(static_cast<int64_t>(2));

  Tensor x = Tensor::FromFloat("", {2, 2}, {3.0f, 1.0f, 0.0f, 2.0f});
  Tensor y = svm.operator()<float>(x, {1.0f, 0.0f, 0.0f, 1.0f}, {2.0f, -1.0f}, {0.5f}, "LINEAR",
                                   0.0f, 0.0f, 0.0f);

  Expect(node, {x}, {y}, "test_cc_svmregressor_linear", {default_opset, opset}, "backend-test",
         registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
