// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/checker.h"
#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

struct InputAttributeCase {
  std::string op;
  int64_t source;
  std::vector<int64_t> shape;
  std::vector<int64_t> parameter;
  std::vector<std::vector<int64_t>> output_shapes;
};

const InputAttributeCase input_attribute_cases[] = {
    {"ReduceSum", 13, {2, 2}, {0}, {{1, 2}}}, {"ReduceMean", 18, {2, 2}, {0}, {{1, 2}}},
    {"Squeeze", 13, {1, 1}, {0}, {{1}}},      {"Unsqueeze", 13, {2}, {0}, {{1, 2}}},
    {"Split", 13, {4}, {2, 2}, {{2}, {2}}},
};

void AddValue(ValueInfoProto *value, const std::string &name, TensorProto::DataType dtype,
              const std::vector<int64_t> &shape) {
  value->set_name(name);
  auto *type = value->mutable_type()->mutable_tensor_type();
  type->set_elem_type(dtype);
  for (int64_t dim : shape) {
    type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
}

ModelProto MakeInputAttributeModel(const InputAttributeCase &test_case, const std::string &provider,
                                   const std::string &sharing = "none") {
  ModelProto model;
  model.set_ir_version(8);
  model.add_opset_import()->set_version(test_case.source);
  auto *graph = model.mutable_graph();
  graph->set_name("input_attribute_safety");
  AddValue(graph->add_input(), "X", TensorProto::FLOAT, test_case.shape);
  const std::vector<int64_t> parameter_shape = {static_cast<int64_t>(test_case.parameter.size())};
  TensorProto parameter;
  parameter.set_name("parameter");
  parameter.set_data_type(TensorProto::INT64);
  *parameter.add_dims() = parameter_shape[0];
  for (int64_t value : test_case.parameter) {
    *parameter.add_int64_data() = value;
  }
  if (provider == "constant") {
    auto *constant = graph->add_node();
    constant->set_op_type("Constant");
    *constant->add_output() = "parameter";
    auto *value = constant->add_attribute();
    value->set_name("value");
    value->set_type(AttributeProto::TENSOR);
    *value->mutable_t() = parameter;
  } else if (provider == "initializer" || provider == "overridable") {
    *graph->add_initializer() = parameter;
  }
  if (provider == "dynamic" || provider == "overridable") {
    AddValue(graph->add_input(), "parameter", TensorProto::INT64, parameter_shape);
  }
  auto *node = graph->add_node();
  node->set_op_type(test_case.op);
  *node->add_input() = "X";
  if (provider != "absent") {
    *node->add_input() = provider == "omitted" ? "" : "parameter";
  }
  for (size_t i = 0; i < test_case.output_shapes.size(); ++i) {
    const std::string name = "Y" + std::to_string(i);
    *node->add_output() = name;
    AddValue(graph->add_output(), name, TensorProto::FLOAT, test_case.output_shapes[i]);
  }
  if (sharing == "node") {
    auto *consumer = graph->add_node();
    consumer->set_op_type("Unique");
    *consumer->add_input() = "parameter";
    *consumer->add_output() = "unique";
    AddValue(graph->add_output(), "unique", TensorProto::INT64, {1});
  } else if (sharing == "output") {
    AddValue(graph->add_output(), "parameter", TensorProto::INT64, parameter_shape);
  }
  return model;
}

class NonconstantAttributeInput
    : public testing::TestWithParam<std::tuple<InputAttributeCase, std::string>> {};

TEST_P(NonconstantAttributeInput, RejectsWithoutChangingSource) {
  const auto &[test_case, provider] = GetParam();
  const ModelProto original = MakeInputAttributeModel(test_case, provider);
  ASSERT_NO_THROW(checker::check_model(original, true));
  const auto before = original.SerializeAsString();
  try {
    version_conversion::ConvertVersion(original, test_case.source - 1);
    FAIL() << "Conversion must reject nonconstant inputs";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("dynamic inputs and overridable initializers"),
              std::string::npos);
  }
  EXPECT_EQ(original.SerializeAsString(), before);
}

INSTANTIATE_TEST_SUITE_P(onnx_version_converter, NonconstantAttributeInput,
                         testing::Combine(testing::ValuesIn(input_attribute_cases),
                                          testing::Values("dynamic", "overridable")));

class ConstantAttributeInput
    : public testing::TestWithParam<std::tuple<InputAttributeCase, std::string, std::string>> {};

TEST_P(ConstantAttributeInput, PreservesValuesAndOtherUses) {
  const auto &[test_case, provider, sharing] = GetParam();
  const ModelProto original = MakeInputAttributeModel(test_case, provider, sharing);
  ASSERT_NO_THROW(checker::check_model(original, true));
  const auto before = original.SerializeAsString();
  const ModelProto converted = version_conversion::ConvertVersion(original, test_case.source - 1);
  EXPECT_EQ(original.SerializeAsString(), before);
  ASSERT_NO_THROW(checker::check_model(converted, true));
  EXPECT_EQ(converted.ref_opset_import()[0].ref_version(), test_case.source - 1);
  const auto &graph = converted.ref_graph();
  ASSERT_EQ(graph.ref_input().size(), 1u);
  const bool retained = sharing != "none";
  size_t constant_count = 0;
  size_t converted_count = 0;
  for (const auto &node : graph.ref_node()) {
    if (node.ref_op_type() == test_case.op) {
      ++converted_count;
      ASSERT_EQ(node.ref_input().size(), 1u);
      EXPECT_EQ(node.ref_input()[0], "X");
      ASSERT_EQ(node.ref_attribute().size(), 1u);
      const auto &attribute = node.ref_attribute()[0];
      EXPECT_EQ(attribute.ref_name(), test_case.op == "Split" ? "split" : "axes");
      EXPECT_EQ(attribute.ref_type(), AttributeProto::INTS);
      ASSERT_EQ(attribute.ref_ints().size(), test_case.parameter.size());
      for (size_t i = 0; i < test_case.parameter.size(); ++i) {
        EXPECT_EQ(attribute.ref_ints()[i], test_case.parameter[i]);
      }
    } else if (node.ref_op_type() == "Constant") {
      ++constant_count;
      EXPECT_EQ(node.SerializeAsString(), original.ref_graph().ref_node()[0].SerializeAsString());
    } else {
      EXPECT_EQ(node.ref_op_type(), "Unique");
      EXPECT_EQ(node.ref_input()[0], "parameter");
      EXPECT_EQ(node.ref_output()[0], "unique");
    }
  }
  EXPECT_EQ(converted_count, 1u);
  EXPECT_EQ(constant_count, retained && provider == "constant" ? 1u : 0u);
  ASSERT_EQ(graph.ref_initializer().size(), retained && provider == "initializer" ? 1u : 0u);
  if (!graph.ref_initializer().empty()) {
    EXPECT_EQ(graph.ref_initializer()[0].SerializeAsString(),
              original.ref_graph().ref_initializer()[0].SerializeAsString());
  }
  ASSERT_EQ(graph.ref_output().size(), test_case.output_shapes.size() + (retained ? 1 : 0));
  if (sharing == "output") {
    EXPECT_EQ(graph.ref_output()[test_case.output_shapes.size()].ref_name(), "parameter");
  }
}

INSTANTIATE_TEST_SUITE_P(onnx_version_converter, ConstantAttributeInput,
                         testing::Combine(testing::ValuesIn(input_attribute_cases),
                                          testing::Values("constant", "initializer"),
                                          testing::Values("none", "node", "output")));

TEST(onnx_version_converter, RemovesOmittedAttributeInputSlot) {
  const InputAttributeCase cases[] = {
      {"ReduceSum", 13, {2, 2}, {}, {{1, 1}}},
      {"ReduceMean", 18, {2, 2}, {}, {{1, 1}}},
      {"Squeeze", 13, {1, 2, 1}, {}, {{2}}},
      {"Split", 13, {4}, {}, {{2}, {2}}},
  };
  for (const auto &test_case : cases) {
    for (const std::string provider : {"omitted", "absent"}) {
      SCOPED_TRACE(test_case.op + "/" + provider);
      const ModelProto original = MakeInputAttributeModel(test_case, provider);
      ASSERT_NO_THROW(checker::check_model(original, true));
      const auto before = original.SerializeAsString();
      const ModelProto converted =
          version_conversion::ConvertVersion(original, test_case.source - 1);
      EXPECT_EQ(original.SerializeAsString(), before);
      ASSERT_NO_THROW(checker::check_model(converted, true));
      ASSERT_EQ(converted.ref_graph().ref_node().size(), 1u);
      const auto &node = converted.ref_graph().ref_node()[0];
      ASSERT_EQ(node.ref_input().size(), 1u);
      EXPECT_EQ(node.ref_input()[0], "X");
      EXPECT_TRUE(node.ref_attribute().empty());
    }
  }
}

} // namespace
