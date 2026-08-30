// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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
void RegisterCategoryMapperCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);

  const std::vector<std::string> cats_strings{"hello", "world", "good morning"};
  const std::vector<int64_t> cats_int64s{1, 2, 3};

  if (mode == TestMode::BENCHMARK) {
    const int64_t count = 65536;
    const std::string default_string = "_Unused";
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
    AttributeProto *default_attr = node.add_attribute();
    default_attr->set_name("default_string");
    default_attr->set_type(AttributeProto::AttributeType::STRING);
    default_attr->set_s(utils::String(default_string));
    Expect(registry, std::move(node), "test_cc_category_mapper_benchmark", {default_opset, opset},
           {count}, {count}, [opset, cats_strings, cats_int64s, default_string]() -> IoData {
             const KernelContext category_mapper_ctx{opset};
             const onnx_kernels::kernel::CategoryMapper category_mapper{category_mapper_ctx};

             std::vector<int64_t> x_values(static_cast<size_t>(count));
             for (int64_t i = 0; i < count; ++i) {
               x_values[static_cast<size_t>(i)] = (i % 3) + 1;
             }
             Tensor x = Tensor::FromInt64("", {count}, x_values);
             Tensor y = category_mapper.template operator()<int64_t, std::string>(
                 x, cats_strings, cats_int64s, default_string);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

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

    Expect(registry, std::move(node), "test_cc_category_mapper_string_to_int",
           {default_opset, opset}, [opset, cats_strings, cats_int64s, default_int64]() -> IoData {
             const KernelContext category_mapper_ctx{opset};
             const onnx_kernels::kernel::CategoryMapper category_mapper{category_mapper_ctx};

             Tensor x = Tensor::FromStrings("", {4}, {"hello", "world", "?", "good morning"});
             Tensor y = category_mapper.template operator()<std::string, int64_t>(
                 x, cats_strings, cats_int64s, default_int64);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
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

    Expect(registry, std::move(node), "test_cc_category_mapper_int_to_string",
           {default_opset, opset}, [opset, default_string, cats_strings, cats_int64s]() -> IoData {
             const KernelContext category_mapper_ctx{opset};
             const onnx_kernels::kernel::CategoryMapper category_mapper{category_mapper_ctx};

             Tensor x = Tensor::FromInt64("", {4}, {1, 2, 4, 3});
             Tensor y = category_mapper.template operator()<int64_t, std::string>(
                 x, cats_strings, cats_int64s, default_string);

             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
