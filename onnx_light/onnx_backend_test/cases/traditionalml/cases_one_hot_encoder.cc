// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"
#include "onnx_kernels/test_case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// OneHotEncoder — for each input element ``x[i]``, emit a one-hot row of
// length ``cats.size()``: column ``k`` is ``1.0`` when ``cats[k] == x[i]``,
// ``0.0`` otherwise. The output is always ``float`` with shape equal to the
// input shape extended by one trailing dimension (``cats.size()``). Since
// opset 1 in the ``ai.onnx.ml`` domain.
// ---------------------------------------------------------------------------
void RegisterOneHotEncoderCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::OneHotEncoder one_hot{ctx};

  // int64 categories with int64 input (canonical "label -> one-hot" case).
  {
    NodeProto node;
    node.set_op_type("OneHotEncoder");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> cats{0, 1, 2, 3};
    AttributeProto *cats_attr = node.add_attribute();
    cats_attr->set_name("cats_int64s");
    cats_attr->set_type(AttributeProto::AttributeType::INTS);
    for (int64_t v : cats) {
      cats_attr->ints().push_back(v);
    }

    AttributeProto *zeros_attr = node.add_attribute();
    zeros_attr->set_name("zeros");
    zeros_attr->set_type(AttributeProto::AttributeType::INT);
    zeros_attr->set_i(static_cast<int64_t>(1));

    Tensor x = Tensor::FromInt64("", {3}, {0, 2, 7});
    Tensor y = one_hot.operator()<int64_t>(x, cats, /*zeros=*/true);

    Expect(node, {x}, {y}, "test_cc_one_hot_encoder_int64", {default_opset, opset}, "backend-test",
           registry);
  }

  // String categories — mirrors the upstream ONNX node test
  // ``test_ai_onnx_ml_one_hot_encoder_string`` style coverage.
  {
    NodeProto node;
    node.set_op_type("OneHotEncoder");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    const std::vector<std::string> cats{"a", "b", "c"};
    AttributeProto *cats_attr = node.add_attribute();
    cats_attr->set_name("cats_strings");
    cats_attr->set_type(AttributeProto::AttributeType::STRINGS);
    for (const std::string &v : cats) {
      *cats_attr->add_strings() = utils::String(v);
    }

    AttributeProto *zeros_attr = node.add_attribute();
    zeros_attr->set_name("zeros");
    zeros_attr->set_type(AttributeProto::AttributeType::INT);
    zeros_attr->set_i(static_cast<int64_t>(1));

    Tensor x = Tensor::FromStrings("", {4}, {"a", "b", "d", "c"});
    Tensor y = one_hot(x, cats, /*zeros=*/true);

    Expect(node, {x}, {y}, "test_cc_one_hot_encoder_string", {default_opset, opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
