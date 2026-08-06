// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterLinearClassifierCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset("ai.onnx.ml", 1);
  const KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const onnx_kernels::kernel::LinearClassifier cls{ctx};

  if (mode == TestMode::BENCHMARK) {
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

    Expect(registry, std::move(node), "test_cc_linearclassifier_int64_binary_benchmark",
           {default_opset, opset}, {16384}, {8192, 16384}, [cls]() -> IoData {
             Tensor x = Tensor::FromFloat("", {8192, 2}, Randn<float>({8192, 2}, 2651));
             auto [y, z] = cls.operator()<float>(x, {1.0f, -1.0f}, {0.0f},
                                                 std::vector<int64_t>{0, 1}, "NONE");
             return IoData{{std::move(x)}, {std::move(y), std::move(z)}};
           });
    return;
  }

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

  Expect(registry, std::move(node), "test_cc_linearclassifier_int64_binary", {default_opset, opset},
         [=]() -> IoData {
           Tensor x = Tensor::FromFloat("", {2, 2}, {2.0f, 1.0f, 0.0f, 3.0f});
           auto yz =
               cls.operator()<float>(x, {1.0f, -1.0f}, {0.0f}, std::vector<int64_t>{0, 1}, "NONE");
           return IoData{{std::move(x)}, {std::move(yz.first), std::move(yz.second)}};
         });
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
