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

void AddFloatValue(ValueInfoProto *value, const std::string &name,
                   const std::vector<int64_t> &shape) {
  value->set_name(name);
  auto *type = value->mutable_type()->mutable_tensor_type();
  type->set_elem_type(TensorProto::FLOAT);
  for (int64_t dim : shape) {
    type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
}

ModelProto MakeUpsampleModel(const std::string &provider, const std::string &sharing) {
  ModelProto model;
  model.set_ir_version(8);
  model.add_opset_import()->set_version(9);
  auto *graph = model.mutable_graph();
  graph->set_name("shared_scales");
  AddFloatValue(graph->add_input(), "X", {1, 1, 2, 2});
  AddFloatValue(graph->add_output(), "Y", {1, 1, 4, 4});

  TensorProto scales;
  scales.set_name("scales");
  scales.set_data_type(TensorProto::FLOAT);
  *scales.add_dims() = 4;
  for (float scale : {1.0f, 1.0f, 2.0f, 2.0f}) {
    *scales.add_float_data() = scale;
  }
  if (provider == "constant") {
    AddFloatValue(graph->add_value_info(), "scales", {4});
    auto *constant = graph->add_node();
    constant->set_op_type("Constant");
    *constant->add_output() = "scales";
    auto *value = constant->add_attribute();
    value->set_name("value");
    value->set_type(AttributeProto::TENSOR);
    *value->mutable_t() = scales;
  } else if (provider != "dynamic") {
    *graph->add_initializer() = scales;
  }
  if (provider == "overridable" || provider == "dynamic") {
    AddFloatValue(graph->add_input(), "scales", {4});
  }
  auto *upsample = graph->add_node();
  upsample->set_op_type("Upsample");
  *upsample->add_input() = "X";
  *upsample->add_input() = "scales";
  *upsample->add_output() = "Y";
  auto *mode = upsample->add_attribute();
  mode->set_name("mode");
  mode->set_type(AttributeProto::STRING);
  mode->set_s("nearest");

  if (sharing == "node") {
    auto *identity = graph->add_node();
    identity->set_op_type("Identity");
    *identity->add_input() = "scales";
    *identity->add_output() = "copy";
    AddFloatValue(graph->add_output(), "copy", {4});
  } else if (sharing == "output") {
    AddFloatValue(graph->add_output(), "scales", {4});
  } else if (sharing == "upsample") {
    NodeProto second = *upsample;
    second.ref_output()[0] = "Z";
    *graph->add_node() = second;
    AddFloatValue(graph->add_output(), "Z", {1, 1, 4, 4});
  }
  return model;
}

class UpsampleSharedScales : public testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(UpsampleSharedScales, PreservesLiveScalesAndRemovesFinalUse) {
  const auto &[provider, sharing] = GetParam();
  const ModelProto original = MakeUpsampleModel(provider, sharing);
  ASSERT_NO_THROW(checker::check_model(original, true));
  const auto before = original.SerializeAsString();
  const ModelProto converted = version_conversion::ConvertVersion(original, 8);
  EXPECT_EQ(original.SerializeAsString(), before);
  ASSERT_NO_THROW(checker::check_model(converted, true));
  ASSERT_EQ(converted.ref_opset_import().size(), 1u);
  EXPECT_EQ(converted.ref_opset_import()[0].ref_version(), 8);
  const auto &graph = converted.ref_graph();
  ASSERT_EQ(graph.ref_input().size(), 1u);
  EXPECT_EQ(graph.ref_input()[0].ref_name(), "X");
  EXPECT_EQ(graph.ref_output().size(), sharing == "none" ? 1u : 2u);

  size_t constant_count = 0;
  size_t upsample_count = 0;
  for (const auto &node : graph.ref_node()) {
    if (node.ref_op_type() == "Upsample") {
      ++upsample_count;
      ASSERT_EQ(node.ref_input().size(), 1u);
      EXPECT_EQ(node.ref_input()[0], "X");
      bool found_scales = false;
      for (const auto &attribute : node.ref_attribute()) {
        if (attribute.ref_name() == "scales") {
          found_scales = true;
          EXPECT_EQ(attribute.ref_type(), AttributeProto::FLOATS);
          ASSERT_EQ(attribute.ref_floats().size(), 4u);
          for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(attribute.ref_floats()[i], i < 2 ? 1.0f : 2.0f);
          }
        }
        if (attribute.ref_name() == "mode") {
          EXPECT_EQ(attribute.ref_s(), "nearest");
        }
      }
      EXPECT_TRUE(found_scales);
    } else if (node.ref_op_type() == "Constant") {
      ++constant_count;
      EXPECT_EQ(node.ref_output()[0], "scales");
      EXPECT_EQ(node.ref_attribute()[0].ref_t().SerializeAsString(),
                original.ref_graph().ref_node()[0].ref_attribute()[0].ref_t().SerializeAsString());
    } else {
      EXPECT_EQ(node.ref_op_type(), "Identity");
      EXPECT_EQ(node.ref_input()[0], "scales");
      EXPECT_EQ(node.ref_output()[0], "copy");
    }
  }
  EXPECT_EQ(upsample_count, sharing == "upsample" ? 2u : 1u);
  const bool retained = sharing == "node" || sharing == "output";
  EXPECT_EQ(constant_count, retained && provider == "constant" ? 1u : 0u);
  ASSERT_EQ(graph.ref_initializer().size(), retained && provider == "initializer" ? 1u : 0u);
  if (!graph.ref_initializer().empty()) {
    EXPECT_EQ(graph.ref_initializer()[0].SerializeAsString(),
              original.ref_graph().ref_initializer()[0].SerializeAsString());
  }
  if (sharing == "output") {
    EXPECT_EQ(graph.ref_output()[1].ref_name(), "scales");
  }
}

INSTANTIATE_TEST_SUITE_P(onnx_version_converter, UpsampleSharedScales,
                         testing::Combine(testing::Values("constant", "initializer"),
                                          testing::Values("none", "node", "output", "upsample")));

class UpsampleNonconstantScales : public testing::TestWithParam<std::string> {};

TEST_P(UpsampleNonconstantScales, RejectsWithoutChangingSource) {
  const ModelProto original = MakeUpsampleModel(GetParam(), "none");
  ASSERT_NO_THROW(checker::check_model(original, true));
  const auto before = original.SerializeAsString();
  try {
    version_conversion::ConvertVersion(original, 8);
    FAIL() << "Conversion must reject nonconstant scales";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what())
                  .find(GetParam() == "overridable" ? "overridable" : "unavailable input"),
              std::string::npos);
  }
  EXPECT_EQ(original.SerializeAsString(), before);
}

INSTANTIATE_TEST_SUITE_P(onnx_version_converter, UpsampleNonconstantScales,
                         testing::Values("overridable", "dynamic"));

} // namespace
