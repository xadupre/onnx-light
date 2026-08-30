// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterLinearRegressorCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("LinearRegressor");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *coefficients = node.add_attribute();
    coefficients->set_name("coefficients");
    coefficients->set_type(AttributeProto::AttributeType::FLOATS);
    coefficients->add_floats(0.5f);
    coefficients->add_floats(-1.0f);

    AttributeProto *intercepts = node.add_attribute();
    intercepts->set_name("intercepts");
    intercepts->set_type(AttributeProto::AttributeType::FLOATS);
    intercepts->add_floats(0.25f);

    AttributeProto *targets = node.add_attribute();
    targets->set_name("targets");
    targets->set_type(AttributeProto::AttributeType::INT);
    targets->set_i(static_cast<int64_t>(1));

    AttributeProto *post_transform = node.add_attribute();
    post_transform->set_name("post_transform");
    post_transform->set_type(AttributeProto::AttributeType::STRING);
    post_transform->set_s("NONE");

    Expect(registry, std::move(node), "test_cc_linearregressor_single_target_benchmark",
           {default_opset, opset}, {16384}, {8192}, [opset]() -> IoData {
             const KernelContext reg_ctx{opset};
             const onnx_kernels::kernel::LinearRegressor reg{reg_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {8192, 2}, 2661);
             Tensor y = reg.template operator()<float>(x, {0.5f, -1.0f}, {0.25f}, 1, "NONE");
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  NodeProto node;
  node.set_op_type("LinearRegressor");
  node.set_domain("ai.onnx.ml");
  node.add_input("x");
  node.add_output("y");

  AttributeProto *coefficients = node.add_attribute();
  coefficients->set_name("coefficients");
  coefficients->set_type(AttributeProto::AttributeType::FLOATS);
  coefficients->add_floats(0.5f);
  coefficients->add_floats(-1.0f);

  AttributeProto *intercepts = node.add_attribute();
  intercepts->set_name("intercepts");
  intercepts->set_type(AttributeProto::AttributeType::FLOATS);
  intercepts->add_floats(0.25f);

  AttributeProto *targets = node.add_attribute();
  targets->set_name("targets");
  targets->set_type(AttributeProto::AttributeType::INT);
  targets->set_i(static_cast<int64_t>(1));

  AttributeProto *post_transform = node.add_attribute();
  post_transform->set_name("post_transform");
  post_transform->set_type(AttributeProto::AttributeType::STRING);
  post_transform->set_s("NONE");

  Expect(registry, std::move(node), "test_cc_linearregressor_single_target", {default_opset, opset},
         [opset]() -> IoData {
           const KernelContext reg_ctx{opset};
           const onnx_kernels::kernel::LinearRegressor reg{reg_ctx};

           Tensor x = Tensor::FromFloat("", {2, 2}, {2.0f, 1.0f, 0.0f, 3.0f});
           Tensor y = reg.template operator()<float>(x, {0.5f, -1.0f}, {0.25f}, 1, "NONE");
           return IoData{{std::move(x)}, {std::move(y)}};
         });
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
