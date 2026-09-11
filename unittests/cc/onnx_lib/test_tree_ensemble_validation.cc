// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"
#include "onnx_extensions/shapes/shapes/traditionalml/shape_traditionalml.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/onnx_tree_ensemble.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

AttributeProto *MutableAttribute(NodeProto &node, const char *name) {
  for (auto &attr : node.attribute()) {
    if (attr.name() == name) {
      return &attr;
    }
  }
  throw std::invalid_argument(std::string("Missing test attribute: ") + name);
}

NodeProto MakeTree() {
  NodeProto node = MakeNode("TreeEnsemble", {"X"}, {"Y"}, "ai.onnx.ml");
  AddAttribute<int64_t>(node, "n_targets", 1);
  for (const auto &[name, values] : {std::pair{"tree_roots", std::vector<int64_t>{0}},
                                     {"nodes_featureids", {0, 0}},
                                     {"nodes_truenodeids", {1, 0}},
                                     {"nodes_falsenodeids", {0, 0}},
                                     {"nodes_trueleafs", {0, 1}},
                                     {"nodes_falseleafs", {1, 1}},
                                     {"leaf_targetids", {0}}}) {
    AddAttribute(node, name, values);
  }
  for (const char *name : {"nodes_splits", "nodes_modes", "leaf_weights"}) {
    auto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::TENSOR);
    auto *tensor = attr->mutable_t();
    const bool modes = std::string(name) == "nodes_modes";
    const int64_t count = std::string(name) == "leaf_weights" ? 1 : 2;
    tensor->set_data_type(modes ? TensorProto::UINT8 : TensorProto::FLOAT);
    tensor->add_dims(count);
    for (int64_t i = 0; i < count; ++i) {
      if (modes) {
        tensor->add_int32_data(0);
      } else {
        tensor->add_float_data(1.0f);
      }
    }
  }
  return node;
}

void CheckTree(const NodeProto &node) {
  checker::CheckerContext ctx;
  ctx.set_ir_version(IR_VERSION);
  ctx.set_opset_imports({{"ai.onnx.ml", 5}});
  checker::check_node(node, ctx, checker::LexicalScopeContext{});
}

void InferTree(NodeProto node) {
  TypeProto input;
  auto *tensor = input.mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape()->add_dim()->set_dim_value(2);
  tensor->mutable_shape()->add_dim()->set_dim_value(1);
  std::unordered_map<std::string, TypeProto *> types{{"X", &input}};
  std::unordered_map<std::string, const TensorProto *> data;
  std::unordered_map<std::string, const SparseTensorProto *> sparse_data;
  ShapeInferenceOptions options;
  shape_inference::InferenceContextImpl ctx(node, types, data, sparse_data, options);
  const auto *schema = OpSchemaRegistry::Schema("TreeEnsemble", 5, "ai.onnx.ml");
  ASSERT_NE(schema, nullptr);
  schema->GetTypeAndShapeInferenceFunction()(ctx);
}

void ComputeTreeShape(const NodeProto &node) {
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                         core::symbolic::SymShape{2, 1}));
  onnx_shapes::shapes::traditionalml::ComputeShapeTreeEnsemble(ctx, node, "X");
}

void BuildTree(const NodeProto &node) {
  core::builder::GraphBuilder builder("tree");
  builder.SetOpsetVersion("ai.onnx.ml", 5);
  builder.ReserveName("X");
  builder.MakeNode("TreeEnsemble", {"X"}, {"Y"}, "ai.onnx.ml", "", node.attribute());
}

void ExpectRejected(const NodeProto &node, const std::string &message) {
  const auto error = ValidateTreeEnsembleAttributes(
      [&node](const char *name) { return FindAttribute(node, name); });
  EXPECT_NE(error.find(message), std::string::npos) << error;
  EXPECT_THROW(CheckTree(node), checker::ValidationError);
  EXPECT_THROW(InferTree(node), InferenceError);
  EXPECT_THROW(ComputeTreeShape(node), std::invalid_argument);
  EXPECT_THROW(BuildTree(node), core::builder::BuilderError);
}

} // namespace

TEST(TreeEnsembleValidation, RejectsCycle) {
  auto node = MakeTree();
  MutableAttribute(node, "nodes_trueleafs")->ints()[1] = 0;
  ExpectRejected(node, "cycle or shared internal node at index 0 in tree 0");
  node = MakeTree();
  MutableAttribute(node, "nodes_truenodeids")->ints()[0] = 0;
  ExpectRejected(node, "cycle or shared internal node at index 0 in tree 0");
}

TEST(TreeEnsembleValidation, RejectsSharedInternalChild) {
  auto node = MakeTree();
  MutableAttribute(node, "nodes_falsenodeids")->ints()[0] = 1;
  MutableAttribute(node, "nodes_falseleafs")->ints()[0] = 0;
  ExpectRejected(node, "cycle or shared internal node at index 1 in tree 0");
}

TEST(TreeEnsembleValidation, ChecksEveryRoot) {
  auto node = MakeTree();
  MutableAttribute(node, "tree_roots")->ints() = {1, 0};
  MutableAttribute(node, "nodes_falsenodeids")->ints()[0] = 1;
  MutableAttribute(node, "nodes_falseleafs")->ints()[0] = 0;
  ExpectRejected(node, "cycle or shared internal node at index 1 in tree 1");
}

TEST(TreeEnsembleValidation, AllowsSharedLeavesAndIndependentRootTraversals) {
  auto node = MakeTree();
  MutableAttribute(node, "tree_roots")->ints() = {0, 1, 0};
  EXPECT_NO_THROW(CheckTree(node));
  EXPECT_NO_THROW(InferTree(node));
  EXPECT_NO_THROW(ComputeTreeShape(node));
  EXPECT_NO_THROW(BuildTree(node));
}

TEST(TreeEnsembleValidation, RejectsInvalidLeafFlags) {
  for (const char *name : {"nodes_trueleafs", "nodes_falseleafs"}) {
    for (int64_t flag : {-1, 2}) {
      auto node = MakeTree();
      MutableAttribute(node, name)->ints()[0] = flag;
      ExpectRejected(node, std::string(name) + " must contain 0 or 1 at index 0");
    }
  }
}

TEST(TreeEnsembleValidation, DefersUnboundFunctionAttributes) {
  auto node = MakeTree();
  auto *attr = MutableAttribute(node, "nodes_truenodeids");
  attr->ints().clear();
  attr->set_ref_attr_name("true_children");
  checker::CheckerContext ctx;
  ctx.set_ir_version(IR_VERSION);
  ctx.set_opset_imports({{"ai.onnx.ml", 5}});
  ctx.set_is_main_graph(false);
  EXPECT_NO_THROW(checker::check_node(node, ctx, checker::LexicalScopeContext{}));
  EXPECT_NO_THROW(BuildTree(node));
}

TEST(TreeEnsembleValidation, RejectsOutOfRangeIndices) {
  for (const char *name : {"tree_roots", "nodes_truenodeids", "nodes_falsenodeids"}) {
    for (int64_t index : {-1, 2}) {
      SCOPED_TRACE(name);
      SCOPED_TRACE(index);
      auto node = MakeTree();
      MutableAttribute(node, name)->ints()[0] = index;
      ExpectRejected(node, std::string("out of range in ") + name + " at index 0");
    }
  }
  auto node = MakeTree();
  MutableAttribute(node, "nodes_truenodeids")->ints()[1] = 1;
  ExpectRejected(node, "leaf index 1 out of range in nodes_truenodeids at index 1");
}

TEST(TreeEnsembleValidation, RejectsInconsistentIntegerArrayLengths) {
  for (const char *name :
       {"nodes_featureids", "nodes_truenodeids", "nodes_falsenodeids", "nodes_trueleafs",
        "nodes_falseleafs", "leaf_targetids", "nodes_missing_value_tracks_true"}) {
    SCOPED_TRACE(name);
    auto node = MakeTree();
    if (std::string(name) == "nodes_missing_value_tracks_true") {
      AddAttribute(node, name, std::vector<int64_t>{0});
    } else {
      MutableAttribute(node, name)->ints().push_back(0);
    }
    ExpectRejected(node, "must have length");
  }
}

TEST(TreeEnsembleValidation, RejectsInconsistentTensorArrayLengths) {
  for (const char *name : {"nodes_splits", "nodes_modes", "leaf_weights", "nodes_hitrates"}) {
    SCOPED_TRACE(name);
    auto node = MakeTree();
    if (std::string(name) == "nodes_hitrates") {
      auto *attr = node.add_attribute();
      attr->CopyFrom(*FindAttribute(node, "nodes_splits"));
      attr->set_name(name);
    }
    auto *tensor = MutableAttribute(node, name)->mutable_t();
    tensor->dims()[0] += 1;
    if (tensor->data_type() == TensorProto::UINT8) {
      tensor->add_int32_data(0);
    } else {
      tensor->add_float_data(1.0f);
    }
    ExpectRejected(node, std::string("attribute '") + name + "' must have length");
  }
}

TEST(TreeEnsembleValidation, AllowsDeepTreesWithoutRecursion) {
  const size_t count = 100000;
  std::vector<int64_t> true_ids(count), false_ids(count, 0), true_leafs(count, 0),
      false_leafs(count, 1), roots{0};
  for (size_t i = 0; i + 1 < count; ++i) {
    true_ids[i] = static_cast<int64_t>(i + 1);
  }
  true_leafs.back() = 1;
  EXPECT_EQ(
      ValidateTreeEnsembleTopology(roots, count, 1, true_ids, false_ids, true_leafs, false_leafs),
      "");
}
