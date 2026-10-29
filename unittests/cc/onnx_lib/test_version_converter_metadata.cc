// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/checker.h"
#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

#include <map>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

void AddTensorValue(ValueInfoProto *value, const std::string &name, int32_t elem_type,
                    bool scalar = false, int64_t dim = 1) {
  value->set_name(name);
  auto *type = value->mutable_type()->mutable_tensor_type();
  type->set_elem_type(elem_type);
  auto *shape = type->mutable_shape();
  if (!scalar) {
    shape->add_dim()->set_dim_value(dim);
  }
}

template <typename Proto> void AddMetadata(Proto *proto, const std::string &value) {
  auto *prop = proto->add_metadata_props();
  prop->set_key("origin");
  prop->set_value(value);
  prop = proto->add_metadata_props();
  prop->set_key("annotation");
  prop->set_value("");
}

template <typename Proto> void ExpectMetadata(const Proto &actual, const Proto &expected) {
  ASSERT_EQ(actual.metadata_props().size(), expected.metadata_props().size());
  for (size_t i = 0; i < expected.metadata_props().size(); ++i) {
    EXPECT_EQ(actual.metadata_props()[i].SerializeAsString(),
              expected.metadata_props()[i].SerializeAsString());
  }
}

ModelProto MakeNestedMetadataModel() {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(14);
  AddMetadata(&model, "model");
  auto *graph = model.mutable_graph();
  graph->set_name("metadata");
  AddTensorValue(graph->add_input(), "cond", TensorProto::BOOL, true);
  AddTensorValue(graph->add_input(), "X", TensorProto::FLOAT);
  AddTensorValue(graph->add_output(), "Y", TensorProto::FLOAT);
  AddMetadata(graph, "graph");
  auto *node = graph->add_node();
  node->set_op_type("If");
  *node->add_input() = "cond";
  *node->add_output() = "Y";
  AddMetadata(node, "node");
  for (const char *name : {"then_branch", "else_branch"}) {
    auto *attr = node->add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::GRAPH);
    auto *branch = attr->mutable_g();
    branch->set_name(name);
    AddTensorValue(branch->add_output(), "B", TensorProto::FLOAT);
    AddMetadata(branch, std::string(name) + "_graph");
    auto *abs = branch->add_node();
    abs->set_op_type("Abs");
    *abs->add_input() = "X";
    *abs->add_output() = "B";
    AddMetadata(abs, std::string(name) + "_node");
  }
  return model;
}

class VersionConverterMetadata : public testing::TestWithParam<int64_t> {};

TEST_P(VersionConverterMetadata, PreservesGraphAndNodeMetadata) {
  const ModelProto original = MakeNestedMetadataModel();
  ASSERT_NO_THROW(checker::check_model(original));
  const auto before = original.SerializeAsString();
  const ModelProto converted = version_conversion::ConvertVersion(original, GetParam());
  ASSERT_NO_THROW(checker::check_model(converted));
  EXPECT_EQ(original.SerializeAsString(), before);
  EXPECT_EQ(converted.opset_import()[0].version(), GetParam());
  ExpectMetadata(converted, original);
  ExpectMetadata(converted.graph(), original.graph());
  ASSERT_EQ(converted.graph().node().size(), 1u);
  const auto &node = converted.graph().node()[0];
  const auto &original_node = original.graph().node()[0];
  ExpectMetadata(node, original_node);
  ASSERT_EQ(node.attribute().size(), 2u);
  for (const auto &attr : node.attribute()) {
    bool found = false;
    for (const auto &original_attr : original_node.attribute()) {
      if (attr.name() == original_attr.name()) {
        found = true;
        ExpectMetadata(attr.g(), original_attr.g());
        ASSERT_EQ(attr.g().node().size(), 1u);
        ExpectMetadata(attr.g().node()[0], original_attr.g().node()[0]);
      }
    }
    EXPECT_TRUE(found);
  }
}

INSTANTIATE_TEST_SUITE_P(onnx_version_converter, VersionConverterMetadata,
                         testing::Values(13, 14, 15));

TEST(onnx_version_converter, PreservesReplacedNodeMetadata) {
  ModelProto original;
  original.set_ir_version(10);
  original.add_opset_import()->set_version(9);
  auto *graph = original.mutable_graph();
  graph->set_name("replacement");
  AddTensorValue(graph->add_input(), "X", TensorProto::FLOAT);
  AddTensorValue(graph->add_output(), "Y", TensorProto::FLOAT, false, 2);
  auto *scales = graph->add_initializer();
  scales->set_name("scales");
  scales->set_data_type(TensorProto::FLOAT);
  *scales->add_dims() = 1;
  *scales->add_float_data() = 2.0f;
  auto *node = graph->add_node();
  node->set_op_type("Upsample");
  *node->add_input() = "X";
  *node->add_input() = "scales";
  *node->add_output() = "Y";
  AddMetadata(node, "upsample");
  ASSERT_NO_THROW(checker::check_model(original));
  const auto before = original.SerializeAsString();
  const ModelProto converted = version_conversion::ConvertVersion(original, 10);
  ASSERT_NO_THROW(checker::check_model(converted));
  EXPECT_EQ(original.SerializeAsString(), before);
  ASSERT_EQ(converted.graph().node().size(), 1u);
  EXPECT_EQ(converted.graph().node()[0].op_type(), "Resize");
  ExpectMetadata(converted.graph().node()[0], *node);
}

TEST(onnx_version_converter, PreservesAdapterMetadataAndUnchangedSourceEntries) {
  for (bool replace : {false, true}) {
    for (bool source_metadata : {false, true}) {
      for (const char *updated_value : {"adapter", ""}) {
        SCOPED_TRACE(replace);
        SCOPED_TRACE(source_metadata);
        SCOPED_TRACE(updated_value);
        ModelProto original;
        original.set_ir_version(10);
        original.add_opset_import()->set_version(13);
        auto *graph = original.mutable_graph();
        graph->set_name("adapter_metadata");
        AddTensorValue(graph->add_input(), "X", TensorProto::FLOAT);
        AddTensorValue(graph->add_output(), "Y", TensorProto::FLOAT);
        auto *node = graph->add_node();
        node->set_op_type("Relu");
        node->add_input("X");
        node->add_output("Y");
        if (source_metadata)
          AddMetadata(node, "source");
        ASSERT_NO_THROW(checker::check_model(original));
        const auto before = original.SerializeAsString();
        version_conversion::DefaultVersionConverter converter;
        converter.registerAdapter(
            "Relu", 13, 14, [replace, updated_value](std::shared_ptr<Graph> graph, Node *node) {
              Node *adapted = node;
              if (replace) {
                adapted = graph->create(node->kind());
                adapted->addInput(node->inputs()[0]);
                node->replaceAllUsesWith(adapted);
                adapted->insertBefore(node);
                node->destroy();
              }
              StringStringEntryProto updated;
              updated.set_key("origin");
              updated.set_value(updated_value);
              StringStringEntryProto added;
              added.set_key("adapter_only");
              added.set_value("created");
              adapted->setMetadataProps({updated, added});
              return adapted;
            });
        const ModelProto converted = converter.convert_version(original, OpSetID(13), OpSetID(14));
        ASSERT_NO_THROW(checker::check_model(converted));
        EXPECT_EQ(original.SerializeAsString(), before);
        ASSERT_EQ(converted.graph().node().size(), 1u);
        std::map<std::string, std::string> metadata;
        for (const auto &prop : converted.graph().node()[0].metadata_props()) {
          EXPECT_TRUE(metadata.emplace(prop.key().value(), prop.value().value()).second);
        }
        EXPECT_EQ(metadata.size(), source_metadata ? 3u : 2u);
        EXPECT_EQ(metadata.at("origin"), updated_value);
        EXPECT_EQ(metadata.at("adapter_only"), "created");
        if (source_metadata) {
          EXPECT_EQ(metadata.at("annotation"), "");
        }
      }
    }
  }
}

} // namespace
