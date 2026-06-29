// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

void RegisterCeluCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(12);
  const kernel::KernelContext ctx{opset};
  const kernel::Celu celu_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Celu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(2.0f);

    Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = celu_kernel(x, 2.0f);
    Expect(node, {x}, {y}, "test_cc_celu", {opset}, "backend-test", registry);
  }

  {
    NodeProto node;
    node.set_op_type("Celu");
    node.add_input("X");
    node.add_output("Y");

    // No alpha attribute: defaults to 1.0.
    Tensor x = Tensor::FromFloat("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = celu_kernel(x);
    Expect(node, {x}, {y}, "test_cc_celu_default", {opset}, "backend-test", registry);
  }

  // Celu-28 opset: required for float16 and bfloat16 type constraints.
  const OpsetId opset28 = DefaultOpset(28);

  // FLOAT16 test case.
  {
    NodeProto node;
    node.set_op_type("Celu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(2.0f);

    Tensor x = kernel::MakeFloat16Tensor("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = celu_kernel(x, 2.0f);
    Expect(node, {x}, {y}, "test_cc_celu_float16", {opset28}, "backend-test", registry);
  }

  // BFLOAT16 test case.
  {
    NodeProto node;
    node.set_op_type("Celu");
    node.add_input("X");
    node.add_output("Y");

    AttributeProto *alpha = node.add_attribute();
    alpha->set_name("alpha");
    alpha->set_type(AttributeProto::FLOAT);
    alpha->set_f(1.0f);

    Tensor x = kernel::MakeBfloat16Tensor("", {2, 3}, {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
    Tensor y = celu_kernel(x, 1.0f);
    Expect(node, {x}, {y}, "test_cc_celu_bfloat16", {opset28}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
