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
// CategoryMapper — converts strings to integers and vice versa via two
// parallel attribute arrays ``cats_strings`` / ``cats_int64s`` (since opset
// 1 in the ``ai.onnx.ml`` domain). The variant — string→int64 vs.
// int64→string — is selected by which ``default_*`` attribute the user
// supplies. Mirrors the upstream ONNX node tests
// ``test_ai_onnx_ml_category_mapper_string_to_int`` and
// ``test_ai_onnx_ml_category_mapper_int_to_string`` (see
// ``onnx/backend/test/case/node/ai_onnx_ml/category_mapper.py``).
// ---------------------------------------------------------------------------
void RegisterCategoryMapperCases(std::vector<TestCase> &registry) {
  const OpsetId opset("ai.onnx.ml", 1);
  const kernel::KernelContext ctx{opset};
  const OpsetId default_opset = DefaultOpset(13);
  const kernel::CategoryMapper category_mapper{ctx};

  const std::vector<std::string> cats_strings{"hello", "world", "good morning"};
  const std::vector<int64_t> cats_int64s{1, 2, 3};

  // string -> int64 variant.
  {
    NodeProto node;
    node.set_op_type("CategoryMapper");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *cats_strings_attr = node.add_attribute();
    cats_strings_attr->set_name("cats_strings");
    cats_strings_attr->set_type(AttributeProto::AttributeType::STRINGS);
    for (const std::string &v : cats_strings) {
      cats_strings_attr->strings().push_back(utils::String(v));
    }

    AttributeProto *cats_int64s_attr = node.add_attribute();
    cats_int64s_attr->set_name("cats_int64s");
    cats_int64s_attr->set_type(AttributeProto::AttributeType::INTS);
    for (int64_t v : cats_int64s) {
      cats_int64s_attr->ints().push_back(v);
    }

    const int64_t default_int64 = -1;
    AttributeProto *default_attr = node.add_attribute();
    default_attr->set_name("default_int64");
    default_attr->set_type(AttributeProto::AttributeType::INT);
    default_attr->set_i(default_int64);

    Tensor x = Tensor::FromStrings("", {4}, {"hello", "world", "?", "good morning"});
    Tensor y = category_mapper.operator()<std::string, int64_t>(x, cats_strings, cats_int64s,
                                                                default_int64);

    Expect(node, {x}, {y}, "test_cc_category_mapper_string_to_int", {default_opset, opset},
           "backend-test", registry);
  }

  // int64 -> string variant.
  {
    NodeProto node;
    node.set_op_type("CategoryMapper");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");

    AttributeProto *cats_strings_attr = node.add_attribute();
    cats_strings_attr->set_name("cats_strings");
    cats_strings_attr->set_type(AttributeProto::AttributeType::STRINGS);
    for (const std::string &v : cats_strings) {
      cats_strings_attr->strings().push_back(utils::String(v));
    }

    AttributeProto *cats_int64s_attr = node.add_attribute();
    cats_int64s_attr->set_name("cats_int64s");
    cats_int64s_attr->set_type(AttributeProto::AttributeType::INTS);
    for (int64_t v : cats_int64s) {
      cats_int64s_attr->ints().push_back(v);
    }

    const std::string default_string = "_Unused";
    AttributeProto *default_attr = node.add_attribute();
    default_attr->set_name("default_string");
    default_attr->set_type(AttributeProto::AttributeType::STRING);
    default_attr->set_s(utils::String(default_string));

    Tensor x = Tensor::FromInt64("", {4}, {1, 2, 4, 3});
    Tensor y = category_mapper.operator()<int64_t, std::string>(x, cats_strings, cats_int64s,
                                                                default_string);

    Expect(node, {x}, {y}, "test_cc_category_mapper_int_to_string", {default_opset, opset},
           "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
