// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_extensions/patterns/traditionalml/tree_ensemble_pattern.h"

#include "onnx_helper.h"
#include "onnx_op/operator_sets.h"

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape InputShape() {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  shape.PushBack(core::symbolic::SymDim(1));
  return shape;
}

void AddInts(utils::RepeatedProtoField<AttributeProto> &attributes, const char *name,
             const std::vector<int64_t> &values) {
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::INTS);
  for (int64_t value : values) {
    attribute.add_ints(value);
  }
}

void AddFloats(utils::RepeatedProtoField<AttributeProto> &attributes, const char *name,
               const std::vector<float> &values) {
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::FLOATS);
  for (float value : values) {
    attribute.add_floats(value);
  }
}

void AddStrings(utils::RepeatedProtoField<AttributeProto> &attributes, const char *name,
                const std::vector<std::string> &values) {
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::STRINGS);
  for (const std::string &value : values) {
    attribute.add_strings(value);
  }
}

void AddInt(utils::RepeatedProtoField<AttributeProto> &attributes, const char *name,
            int64_t value) {
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(value);
}

void AddString(utils::RepeatedProtoField<AttributeProto> &attributes, const char *name,
               const std::string &value) {
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::STRING);
  attribute.set_s(value);
}

utils::RepeatedProtoField<AttributeProto>
BaseTreeAttributes(const std::string &post_transform = "NONE") {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AddInts(attributes, "nodes_treeids", {0, 0, 0});
  AddInts(attributes, "nodes_nodeids", {0, 1, 2});
  AddInts(attributes, "nodes_featureids", {0, 0, 0});
  AddFloats(attributes, "nodes_values", {0.5f, 0.0f, 0.0f});
  AddStrings(attributes, "nodes_modes", {"BRANCH_LEQ", "LEAF", "LEAF"});
  AddInts(attributes, "nodes_truenodeids", {1, 0, 0});
  AddInts(attributes, "nodes_falsenodeids", {2, 0, 0});
  AddInts(attributes, "nodes_missing_value_tracks_true", {1, 0, 0});
  AddString(attributes, "post_transform", post_transform);
  return attributes;
}

void AddRegressor(core::builder::GraphBuilder &builder, const std::string &post_transform = "NONE",
                  const std::string &aggregate_function = "SUM") {
  utils::RepeatedProtoField<AttributeProto> attributes = BaseTreeAttributes(post_transform);
  AddInts(attributes, "target_treeids", {0, 0});
  AddInts(attributes, "target_nodeids", {1, 2});
  AddInts(attributes, "target_ids", {0, 0});
  AddFloats(attributes, "target_weights", {1.0f, 2.0f});
  AddInt(attributes, "n_targets", 1);
  AddString(attributes, "aggregate_function", aggregate_function);
  builder.MakeNode("TreeEnsembleRegressor", {"X"}, {"Y"}, "ai.onnx.ml", "regressor", attributes);
}

void AddClassifier(core::builder::GraphBuilder &builder, bool string_labels,
                   bool base_values = false) {
  utils::RepeatedProtoField<AttributeProto> attributes = BaseTreeAttributes();
  AddInts(attributes, "class_treeids", {0, 0});
  AddInts(attributes, "class_nodeids", {1, 2});
  AddInts(attributes, "class_ids", {0, 1});
  AddFloats(attributes, "class_weights", {1.0f, 2.0f});
  if (string_labels) {
    AddStrings(attributes, "classlabels_strings", {"left", "right"});
  } else {
    AddInts(attributes, "classlabels_int64s", {10, 20});
  }
  if (base_values) {
    AddFloats(attributes, "base_values", {0.25f, -0.5f});
  }
  builder.MakeNode("TreeEnsembleClassifier", {"X"}, {"Y", "Z"}, "ai.onnx.ml", "classifier",
                   attributes);
}

void Optimize(core::builder::GraphBuilder &builder) {
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::TreeEnsemblePattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();
}

std::vector<int64_t> AttributeInts(const NodeProto &node, const char *name) {
  const AttributeProto *attribute = FindAttribute(node, name);
  EXPECT_NE(attribute, nullptr);
  return attribute == nullptr
             ? std::vector<int64_t>()
             : std::vector<int64_t>(attribute->ints().begin(), attribute->ints().end());
}

std::vector<std::string> AttributeStrings(const NodeProto &node, const char *name) {
  const AttributeProto *attribute = FindAttribute(node, name);
  EXPECT_NE(attribute, nullptr);
  std::vector<std::string> values;
  if (attribute != nullptr) {
    values.assign(attribute->strings().begin(), attribute->strings().end());
  }
  return values;
}

std::vector<double> AttributeTensorFloats(const NodeProto &node, const char *name) {
  const AttributeProto *attribute = FindAttribute(node, name);
  EXPECT_NE(attribute, nullptr);
  std::vector<double> values;
  if (attribute != nullptr) {
    EXPECT_TRUE(ReadFloatingValues(attribute->ref_t(), values));
  }
  return values;
}

std::vector<int64_t> AttributeTensorInts(const NodeProto &node, const char *name,
                                         TensorProto::DataType type) {
  const AttributeProto *attribute = FindAttribute(node, name);
  EXPECT_NE(attribute, nullptr);
  std::vector<int64_t> values;
  if (attribute != nullptr) {
    EXPECT_EQ(attribute->ref_t().data_type(), type);
    EXPECT_TRUE(ReadIntegerValues(attribute->ref_t(), values));
  }
  return values;
}

const TensorProto *FindInitializer(const core::builder::GraphBuilder &builder,
                                   TensorProto::DataType type) {
  for (const TensorProto &initializer : builder.Initializers()) {
    if (initializer.data_type() == type) {
      return &initializer;
    }
  }
  return nullptr;
}

} // namespace

TEST(TreeEnsemblePattern, ConvertsRegressorStructureAndAttributes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  AddRegressor(builder);
  builder.MakeOutput("Y");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  const NodeProto &tree = builder.Nodes()[0];
  EXPECT_EQ(tree.op_type().value(), "TreeEnsemble");
  EXPECT_EQ(tree.domain().value(), "ai.onnx.ml");
  EXPECT_EQ(tree.input()[0].value(), "X");
  EXPECT_EQ(tree.output()[0].value(), "Y");
  EXPECT_EQ(builder.OpsetVersion("ai.onnx.ml"), 5);
  EXPECT_EQ(AttributeInts(tree, "tree_roots"), std::vector<int64_t>({0}));
  EXPECT_EQ(AttributeInts(tree, "nodes_featureids"), std::vector<int64_t>({0}));
  EXPECT_EQ(AttributeInts(tree, "nodes_truenodeids"), std::vector<int64_t>({0}));
  EXPECT_EQ(AttributeInts(tree, "nodes_falsenodeids"), std::vector<int64_t>({1}));
  EXPECT_EQ(AttributeInts(tree, "nodes_trueleafs"), std::vector<int64_t>({1}));
  EXPECT_EQ(AttributeInts(tree, "nodes_falseleafs"), std::vector<int64_t>({1}));
  EXPECT_EQ(AttributeInts(tree, "nodes_missing_value_tracks_true"), std::vector<int64_t>({1}));
  EXPECT_EQ(AttributeInts(tree, "leaf_targetids"), std::vector<int64_t>({0, 0}));
  EXPECT_EQ(AttributeTensorFloats(tree, "nodes_splits"), std::vector<double>({0.5}));
  EXPECT_EQ(AttributeTensorInts(tree, "nodes_modes", TensorProto::DataType::UINT8),
            std::vector<int64_t>({0}));
  EXPECT_EQ(AttributeTensorFloats(tree, "leaf_weights"), std::vector<double>({1.0, 2.0}));
  EXPECT_EQ(FindAttribute(tree, "aggregate_function")->i(), 1);
  EXPECT_EQ(FindAttribute(tree, "post_transform")->i(), 0);
}

TEST(TreeEnsemblePattern, ConvertsClassifierWithIntLabels) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  AddClassifier(builder, false);
  builder.MakeOutput("Y");
  builder.MakeOutput("Z");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsemble");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "Z");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "ArgMax");
  EXPECT_EQ(builder.Nodes()[1].input()[0].value(), "Z");
  EXPECT_EQ(FindAttribute(builder.Nodes()[1], "axis")->i(), 1);
  EXPECT_EQ(FindAttribute(builder.Nodes()[1], "keepdims")->i(), 0);
  EXPECT_EQ(builder.Nodes()[2].op_type().value(), "Gather");
  EXPECT_EQ(builder.Nodes()[2].output()[0].value(), "Y");
  const TensorProto *labels = FindInitializer(builder, TensorProto::DataType::INT64);
  ASSERT_NE(labels, nullptr);
  EXPECT_EQ(std::vector<int64_t>(labels->int64_data().begin(), labels->int64_data().end()),
            std::vector<int64_t>({10, 20}));
}

TEST(TreeEnsemblePattern, ConvertsClassifierWithStringLabels) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  AddClassifier(builder, true);
  builder.MakeOutput("Y");
  builder.MakeOutput("Z");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsemble");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "Z");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "ArgMax");
  EXPECT_EQ(builder.Nodes()[1].input()[0].value(), "Z");
  const NodeProto &encoder = builder.Nodes()[2];
  EXPECT_EQ(encoder.op_type().value(), "LabelEncoder");
  EXPECT_EQ(encoder.domain().value(), "ai.onnx.ml");
  EXPECT_EQ(encoder.input()[0].value(), builder.Nodes()[1].output()[0].value());
  EXPECT_EQ(encoder.output()[0].value(), "Y");
  EXPECT_EQ(AttributeInts(encoder, "keys_int64s"), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(AttributeStrings(encoder, "values_strings"),
            std::vector<std::string>({"left", "right"}));
  EXPECT_EQ(FindInitializer(builder, TensorProto::DataType::INT64), nullptr);
}

TEST(TreeEnsemblePattern, InsertsBaseValuesAdd) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  AddClassifier(builder, false, true);
  builder.MakeOutput("Y");
  builder.MakeOutput("Z");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 4u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsemble");
  EXPECT_NE(builder.Nodes()[0].output()[0].value(), "Z");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Add");
  EXPECT_EQ(builder.Nodes()[1].output()[0].value(), "Z");
  const TensorProto *base = FindInitializer(builder, TensorProto::DataType::FLOAT);
  ASSERT_NE(base, nullptr);
  std::vector<double> values;
  ASSERT_TRUE(ReadFloatingValues(*base, values));
  EXPECT_EQ(values, std::vector<double>({0.25, -0.5}));
}

TEST(TreeEnsemblePattern, AddsRegressorBaseValuesToOriginalOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  utils::RepeatedProtoField<AttributeProto> attributes = BaseTreeAttributes();
  AddInts(attributes, "target_treeids", {0, 0});
  AddInts(attributes, "target_nodeids", {1, 2});
  AddInts(attributes, "target_ids", {0, 0});
  AddFloats(attributes, "target_weights", {1.0f, 2.0f});
  AddFloats(attributes, "base_values", {0.75f});
  AddInt(attributes, "n_targets", 1);
  AddString(attributes, "aggregate_function", "SUM");
  builder.MakeNode("TreeEnsembleRegressor", {"X"}, {"Y"}, "ai.onnx.ml", "regressor", attributes);
  builder.MakeOutput("Y");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsemble");
  EXPECT_NE(builder.Nodes()[0].output()[0].value(), "Y");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Add");
  EXPECT_EQ(builder.Nodes()[1].output()[0].value(), "Y");
}

TEST(TreeEnsemblePattern, DuplicatesTreesAndSumsDuplicateWeights) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  utils::RepeatedProtoField<AttributeProto> attributes = BaseTreeAttributes();
  AddInts(attributes, "target_treeids", {0, 0, 0});
  AddInts(attributes, "target_nodeids", {1, 1, 2});
  AddInts(attributes, "target_ids", {0, 0, 1});
  AddFloats(attributes, "target_weights", {1.0f, 2.0f, 4.0f});
  AddInt(attributes, "n_targets", 2);
  AddString(attributes, "aggregate_function", "SUM");
  builder.MakeNode("TreeEnsembleRegressor", {"X"}, {"Y"}, "ai.onnx.ml", "multi", attributes);
  builder.MakeOutput("Y");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  const NodeProto &tree = builder.Nodes()[0];
  EXPECT_EQ(AttributeInts(tree, "tree_roots"), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(AttributeInts(tree, "nodes_featureids"), std::vector<int64_t>({0, 0}));
  EXPECT_EQ(AttributeInts(tree, "leaf_targetids"), std::vector<int64_t>({0, 0, 1, 1}));
  EXPECT_EQ(AttributeTensorFloats(tree, "leaf_weights"), std::vector<double>({3.0, 0.0, 0.0, 4.0}));
}

TEST(TreeEnsemblePattern, ConvertsDegenerateLeafTreeWithDummyNode) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  utils::RepeatedProtoField<AttributeProto> attributes;
  AddInts(attributes, "nodes_treeids", {0});
  AddInts(attributes, "nodes_nodeids", {7});
  AddInts(attributes, "nodes_featureids", {0});
  AddFloats(attributes, "nodes_values", {0.0f});
  AddStrings(attributes, "nodes_modes", {"LEAF"});
  AddInts(attributes, "nodes_truenodeids", {0});
  AddInts(attributes, "nodes_falsenodeids", {0});
  AddInts(attributes, "target_treeids", {0});
  AddInts(attributes, "target_nodeids", {7});
  AddInts(attributes, "target_ids", {0});
  AddFloats(attributes, "target_weights", {3.5f});
  AddInt(attributes, "n_targets", 1);
  AddString(attributes, "aggregate_function", "SUM");
  AddString(attributes, "post_transform", "NONE");
  builder.MakeNode("TreeEnsembleRegressor", {"X"}, {"Y"}, "ai.onnx.ml", "leaf", attributes);
  builder.MakeOutput("Y");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  const NodeProto &tree = builder.Nodes()[0];
  EXPECT_EQ(AttributeInts(tree, "tree_roots"), std::vector<int64_t>({0}));
  EXPECT_EQ(AttributeInts(tree, "nodes_featureids"), std::vector<int64_t>({0}));
  EXPECT_EQ(AttributeInts(tree, "nodes_truenodeids"), std::vector<int64_t>({0}));
  EXPECT_EQ(AttributeInts(tree, "nodes_falsenodeids"), std::vector<int64_t>({0}));
  EXPECT_EQ(AttributeInts(tree, "nodes_trueleafs"), std::vector<int64_t>({1}));
  EXPECT_EQ(AttributeInts(tree, "nodes_falseleafs"), std::vector<int64_t>({1}));
  EXPECT_EQ(AttributeTensorFloats(tree, "leaf_weights"), std::vector<double>({3.5}));
}

TEST(TreeEnsemblePattern, LeavesUnsupportedCandidatesUnchanged) {
  struct UnsupportedCase {
    core::symbolic::TensorType input_type;
    std::string post_transform;
    std::string aggregate_function;
  };
  const std::vector<UnsupportedCase> cases = {
      {core::symbolic::TensorType::kFloat, "LOGISTIC", "SUM"},
      {core::symbolic::TensorType::kFloat, "NONE", "AVERAGE"},
      {core::symbolic::TensorType::kDouble, "NONE", "SUM"},
  };
  for (const UnsupportedCase &test_case : cases) {
    core::builder::GraphBuilder builder("g", SchemaLookup());
    builder.SetOpsetVersion("", 13);
    builder.SetOpsetVersion("ai.onnx.ml", 5);
    builder.MakeInput("X", test_case.input_type, InputShape());
    AddRegressor(builder, test_case.post_transform, test_case.aggregate_function);
    builder.MakeOutput("Y");

    Optimize(builder);

    ASSERT_EQ(builder.Nodes().size(), 1u);
    EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsembleRegressor");
    EXPECT_EQ(builder.OpsetVersion("ai.onnx.ml"), 5);
  }
}

TEST(TreeEnsemblePattern, LeavesInvalidBranchReferencesUnchanged) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  utils::RepeatedProtoField<AttributeProto> attributes = BaseTreeAttributes();
  for (AttributeProto &attribute : attributes) {
    if (attribute.name().value() == "nodes_truenodeids") {
      attribute.ints()[0] = 99;
    }
  }
  AddInts(attributes, "target_treeids", {0, 0});
  AddInts(attributes, "target_nodeids", {1, 2});
  AddInts(attributes, "target_ids", {0, 0});
  AddFloats(attributes, "target_weights", {1.0f, 2.0f});
  AddInt(attributes, "n_targets", 1);
  AddString(attributes, "aggregate_function", "SUM");
  builder.MakeNode("TreeEnsembleRegressor", {"X"}, {"Y"}, "ai.onnx.ml", "invalid", attributes);
  builder.MakeOutput("Y");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsembleRegressor");
}

TEST(TreeEnsemblePattern, LeavesBinaryClassifierWithImplicitSecondScoreUnchanged) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  utils::RepeatedProtoField<AttributeProto> attributes = BaseTreeAttributes();
  AddInts(attributes, "class_treeids", {0, 0});
  AddInts(attributes, "class_nodeids", {1, 2});
  AddInts(attributes, "class_ids", {1, 1});
  AddFloats(attributes, "class_weights", {0.25f, 0.75f});
  AddInts(attributes, "classlabels_int64s", {0, 1});
  builder.MakeNode("TreeEnsembleClassifier", {"X"}, {"Y", "Z"}, "ai.onnx.ml", "binary", attributes);
  builder.MakeOutput("Y");
  builder.MakeOutput("Z");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsembleClassifier");
}

TEST(TreeEnsemblePattern, LeavesRankOneInputUnchanged) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(1));
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, shape);
  AddRegressor(builder);
  builder.MakeOutput("Y");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsembleRegressor");
}

TEST(TreeEnsemblePattern, LeavesOlderMlOpsetUnchanged) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.SetOpsetVersion("ai.onnx.ml", 1);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  AddRegressor(builder);
  builder.MakeOutput("Y");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsembleRegressor");
  EXPECT_EQ(builder.OpsetVersion("ai.onnx.ml"), 1);
}

TEST(TreeEnsemblePattern, LeavesClassifierWithoutDefaultOpsetUnchanged) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  AddClassifier(builder, false);
  builder.MakeOutput("Y");
  builder.MakeOutput("Z");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsembleClassifier");
}

TEST(TreeEnsemblePattern, LeavesBaseValuesWithLegacyAddBroadcastingUnchanged) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 6);
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, InputShape());
  AddClassifier(builder, false, true);
  builder.MakeOutput("Y");
  builder.MakeOutput("Z");

  Optimize(builder);

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "TreeEnsembleClassifier");
}

TEST(TreeEnsemblePattern, CreatesPatternFromRegistry) {
  std::unique_ptr<core::builder::PatternOptimization> pattern =
      onnx_patterns::CreatePattern("TreeEnsemble");
  ASSERT_NE(pattern, nullptr);
  EXPECT_EQ(pattern->Name(), "TreeEnsemble");
  EXPECT_EQ(pattern->FastOpType(),
            std::set<std::string>({"TreeEnsembleClassifier", "TreeEnsembleRegressor"}));
}

} // namespace Test
