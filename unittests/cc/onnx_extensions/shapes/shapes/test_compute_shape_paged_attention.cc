// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"
#include "onnx_proto/onnx_verify.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

TypeProto PagedTensor(std::initializer_list<int64_t> dims) {
  TypeProto type;
  auto *tensor = type.mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape();
  for (int64_t dim : dims) {
    auto *dimension = tensor->mutable_shape()->add_dim();
    if (dim >= 0)
      dimension->set_dim_value(dim);
  }
  return type;
}

TypeProto PagedCache() {
  TypeProto type;
  auto *blocks = type.mutable_struct_type()->mutable_structure()->add_field();
  blocks->set_name("blocks");
  auto *page = blocks->mutable_type()
                   ->mutable_sequence_type()
                   ->mutable_elem_type()
                   ->mutable_struct_type()
                   ->mutable_structure();
  for (const char *name : {"start", "length", "key", "value"}) {
    auto *field = page->add_field();
    field->set_name(name);
    if (std::string(name) == "start" || std::string(name) == "length") {
      *field->mutable_type() = PagedTensor({});
      field->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
    } else
      *field->mutable_type() = PagedTensor({1, 1, -1, -1});
  }
  return type;
}

NodeProto PagedNode() {
  NodeProto node;
  node.set_domain("onnx_light");
  node.set_op_type("PagedAttention");
  for (const char *input : {"Q", "K", "V", "past"})
    node.add_input(input);
  node.add_output("Y");
  node.add_output("present");
  return node;
}

core::shapes::ShapesContext PagedContext() {
  onnx_shapes::RegisterShapeFunctions();
  core::shapes::ShapesContext context;
  context.SetOpsetVersion("onnx_light", 1);
  context.SetType("Q", PagedTensor({1, 1, 3, 4}));
  context.SetType("K", PagedTensor({1, 1, 3, 4}));
  context.SetType("V", PagedTensor({1, 1, 3, 6}));
  context.SetType("past", PagedCache());
  return context;
}

} // namespace

TEST(PagedAttentionShape, DispatchesAndPreservesCacheDeclaration) {
  auto context = PagedContext();
  context.ComputeShapeNode(PagedNode());
  EXPECT_TRUE(context.GetType("Y").Equals(PagedTensor({1, 1, 3, 6})));
  EXPECT_TRUE(context.GetType("present").Equals(PagedCache()));
}

TEST(PagedAttentionShape, PropagatesSymbolsAndSupportsUnknownRank) {
  auto context = PagedContext();
  auto query = PagedTensor({1, 1, -1, 4});
  query.mutable_tensor_type()->mutable_shape()->mutable_dim(2)->set_dim_param("L");
  auto value = PagedTensor({1, 1, -1, -1});
  value.mutable_tensor_type()->mutable_shape()->mutable_dim(3)->set_dim_param("Dv");
  TypeProto key;
  key.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  context.SetType("Q", query);
  context.SetType("K", key);
  context.SetType("V", value);
  context.ComputeShapeNode(PagedNode());
  const auto &output = context.GetType("Y").tensor_type().shape();
  EXPECT_EQ(output.dim(2).dim_param(), "L");
  EXPECT_EQ(output.dim(3).dim_param(), "Dv");
}

TEST(PagedAttentionShape, UnknownInputsStillInferRankFourAndFloat) {
  auto context = PagedContext();
  TypeProto unknown;
  unknown.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  for (const char *name : {"Q", "K", "V"})
    context.SetType(name, unknown);
  context.ComputeShapeNode(PagedNode());
  const auto &output = context.GetType("Y").tensor_type();
  EXPECT_EQ(output.elem_type(), TensorProto::FLOAT);
  ASSERT_EQ(output.shape().dim_size(), 4);
  EXPECT_EQ(output.shape().dim(0).dim_value(), 1);
  EXPECT_EQ(output.shape().dim(1).dim_value(), 1);
  EXPECT_FALSE(output.shape().dim(2).has_dim_value());
  EXPECT_FALSE(output.shape().dim(3).has_dim_value());
}

TEST(PagedAttentionShape, AcceptsSymbolicTensorDescriptorsAndEmptyAppend) {
  auto context = PagedContext();
  for (const char *name : {"Q", "K", "V"})
    context.Set(name, core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                                core::symbolic::SymShape{1, 1, 0, 4}));
  context.ComputeShapeNode(PagedNode());
  EXPECT_TRUE(context.GetType("Y").Equals(PagedTensor({1, 1, 0, 4})));
}

TEST(PagedAttentionShape, RefinesUnknownValueWidthFromCache) {
  auto context = PagedContext();
  auto cache = PagedCache();
  auto *page = cache.mutable_struct_type()
                   ->mutable_structure()
                   ->mutable_field(0)
                   ->mutable_type()
                   ->mutable_sequence_type()
                   ->mutable_elem_type()
                   ->mutable_struct_type()
                   ->mutable_structure();
  page->mutable_field(3)
      ->mutable_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_dim(3)
      ->set_dim_value(6);
  context.SetType("past", cache);
  context.SetType("V", PagedTensor({1, 1, 3, -1}));
  context.ComputeShapeNode(PagedNode());
  EXPECT_TRUE(context.GetType("Y").Equals(PagedTensor({1, 1, 3, 6})));
  EXPECT_TRUE(context.GetType("present").Equals(cache));
}

TEST(PagedAttentionShape, RejectsKnownTensorMismatches) {
  for (const auto &dims : {std::vector<int64_t>{1, 3, 4},
                           {2, 1, 3, 4},
                           {1, 2, 3, 4},
                           {1, 1, 2, 4},
                           {1, 1, 3, 5},
                           {1, 1, 3, 0}}) {
    auto context = PagedContext();
    TypeProto query;
    auto *tensor = query.mutable_tensor_type();
    tensor->set_elem_type(TensorProto::FLOAT);
    for (int64_t dim : dims)
      tensor->mutable_shape()->add_dim()->set_dim_value(dim);
    context.SetType("Q", query);
    EXPECT_THROW(context.ComputeShapeNode(PagedNode()), std::invalid_argument);
    EXPECT_FALSE(context.HasType("Y"));
  }
  auto context = PagedContext();
  auto value = PagedTensor({1, 1, 3, 6});
  value.mutable_tensor_type()->set_elem_type(TensorProto::DOUBLE);
  context.SetType("V", value);
  EXPECT_THROW(context.ComputeShapeNode(PagedNode()), std::invalid_argument);
}

TEST(PagedAttentionShape, RejectsMalformedCacheAndNode) {
  for (int failure = 0; failure < 5; ++failure) {
    auto context = PagedContext();
    auto cache = PagedCache();
    auto *blocks = cache.mutable_struct_type()->mutable_structure()->mutable_field(0);
    auto *page = blocks->mutable_type()
                     ->mutable_sequence_type()
                     ->mutable_elem_type()
                     ->mutable_struct_type()
                     ->mutable_structure();
    if (failure == 0)
      blocks->set_name("wrong");
    else if (failure == 1)
      page->mutable_field(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
          TensorProto::FLOAT);
    else if (failure == 2)
      page->mutable_field(2)
          ->mutable_type()
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(3)
          ->set_dim_value(8);
    else if (failure == 3)
      page->mutable_field(3)->set_name("wrong");
    else
      *blocks->mutable_type() = PagedTensor({1});
    context.SetType("past", cache);
    EXPECT_THROW(context.ComputeShapeNode(PagedNode()), std::invalid_argument);
  }
  auto context = PagedContext();
  auto node = PagedNode();
  node.clear_input();
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapePagedAttention(context, node),
               std::invalid_argument);
  node = PagedNode();
  node.clear_output();
  node.add_output("Y");
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapePagedAttention(context, node),
               std::invalid_argument);
}

TEST(PagedAttentionShape, ModelInferencePreservesReferencesAndFeedback) {
  auto context = PagedContext();
  ModelProto model;
  model.set_ir_version(13);
  auto *opset = model.add_opset_import();
  opset->set_domain("onnx_light");
  opset->set_version(1);
  auto cache = PagedCache();
  auto *definition = model.add_struct_types();
  *definition = cache.struct_type();
  definition->set_type_id(1);
  auto *page = definition->mutable_structure()
                   ->mutable_field(0)
                   ->mutable_type()
                   ->mutable_sequence_type()
                   ->mutable_elem_type();
  StructTypeProto page_definition = page->struct_type();
  page_definition.set_type_id(2);
  *page = TypeProto{};
  page->mutable_struct_type()->set_type_ref(2);
  *model.add_struct_types() = page_definition;
  TypeProto reference;
  reference.mutable_struct_type()->set_type_ref(1);
  auto *graph = model.mutable_graph();
  for (const char *name : {"Q", "K", "V", "past"}) {
    auto *input = graph->add_input();
    input->set_name(name);
    *input->mutable_type() = std::string(name) == "past" ? reference : context.GetType(name);
  }
  *graph->add_node() = PagedNode();
  graph->add_output()->set_name("Y");
  auto *output = graph->add_output();
  output->set_name("present");
  *output->mutable_type() = reference;
  auto *binding = graph->add_persistent_bindings();
  binding->set_input_name("past");
  binding->set_output_name("present");
  context.Clear();
  context.ComputeShapeModel(model);
  context.ApplyInferredShapesToModel(model);
  EXPECT_TRUE(model.graph().output(0).type().Equals(PagedTensor({1, 1, 3, 6})));
  EXPECT_TRUE(model.graph().output(1).type().Equals(reference));
  StructTypeCatalogue catalogue;
  catalogue.Build(model);
  EXPECT_NO_THROW(VerifyPersistentBindings(&catalogue, model.graph()));
}

} // namespace Test
