// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <utility>

#include "onnx_core/shapes/shapes_context.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

TypeProto LogicalTensor() {
  TypeProto type;
  type.ref_tensor_type().set_elem_type(TensorProto::FLOAT);
  type.ref_tensor_type().ref_shape().add_dim()->set_dim_value(8);
  return type;
}

ModelProto StructuredModel() {
  ModelProto model;
  auto *scalar = model.ref_struct_types().Add();
  scalar->set_type_id(uint64_t(1));
  auto &array = scalar->ref_array();
  array.set_dimension(uint64_t(2));
  array.ref_element_type().ref_tensor_type().set_elem_type(TensorProto::UINT8);
  array.ref_element_type().ref_tensor_type().ref_shape();
  auto *record = model.ref_struct_types().Add();
  record->set_type_id(uint64_t(2));
  auto *field = record->ref_structure().add_field();
  field->set_name("codes");
  field->ref_type().ref_struct_type().set_type_ref(uint64_t(1));
  auto *init = model.ref_graph().ref_encoded_initializer().Add();
  init->set_name("encoded");
  init->ref_struct_type().set_type_ref(uint64_t(2));
  init->set_raw_data(std::string(4, '\1'));
  init->ref_logical_type() = LogicalTensor();
  auto *identity = model.ref_graph().add_node();
  identity->set_op_type("Identity");
  identity->add_input("encoded");
  identity->add_output("output");
  model.ref_graph().add_output()->set_name("output");
  return model;
}

NodeProto UnaryNode(const char *op, const char *input, const char *output) {
  NodeProto node;
  node.set_op_type(op);
  node.add_input(input);
  node.add_output(output);
  return node;
}

GraphProto Branch(const char *output) {
  GraphProto graph;
  *graph.add_node() = UnaryNode("Identity", "encoded", output);
  graph.add_output()->set_name(output);
  return graph;
}

NodeProto IfNode(const GraphProto &left, const GraphProto &right) {
  NodeProto node;
  node.set_op_type("If");
  node.add_input("condition");
  node.add_output("result");
  auto *attribute = node.add_attribute();
  attribute->set_name("then_branch");
  attribute->set_type(AttributeProto::GRAPH);
  attribute->set_g(left);
  attribute = node.add_attribute();
  attribute->set_name("else_branch");
  attribute->set_type(AttributeProto::GRAPH);
  attribute->set_g(right);
  return node;
}

} // namespace

TEST(StructuredInference, NestedReferencesLogicalAndPhysicalIdentity) {
  ModelProto model = StructuredModel();
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  ASSERT_TRUE(context.HasType("output"));
  EXPECT_EQ(context.GetType("output").ref_struct_type().ref_type_ref(), 2u);
  ASSERT_TRUE(context.Has("output"));
  EXPECT_EQ(context.Get("output").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(context.Get("output").Shape()[0].AsInt(), 8);
  EXPECT_EQ(context.Get("output").Data(), nullptr);
  EXPECT_FALSE(context.Get("output").HasValueAsShape());
  auto layout = context.GetEncodedLayout("output");
  EXPECT_EQ(layout.element_bits, 16u);
  EXPECT_EQ(layout.payload_bytes, 4u);
  EXPECT_EQ(layout.record_count, 2u);
  ASSERT_NE(layout.root, nullptr);
  EXPECT_EQ(layout.root->ref_type_id(), 2u);
  context.ApplyInferredShapesToModel(model);
  EXPECT_TRUE(model.graph().output(0).type().has_struct_type());
  EXPECT_EQ(model.graph().output(0).type().ref_struct_type().ref_type_ref(), 2u);
}

TEST(StructuredInference, CatalogueAndValuesOwnTheirCopies) {
  core::shapes::ShapesContext copy;
  {
    ModelProto model = StructuredModel();
    core::shapes::ShapesContext original;
    original.ComputeShapeModel(model);
    copy = original;
    model.ref_struct_types().Clear();
    model.ref_graph().ref_encoded_initializer().Clear();
    original.Clear();
    EXPECT_EQ(original.StructTypes().size(), 2u);
    EXPECT_FALSE(original.HasEncodedValue("output"));
  }
  const auto layout = copy.GetEncodedLayout("output");
  EXPECT_EQ(layout.root->ref_type_id(), 2u);
  EXPECT_EQ(layout.payload_bytes, 4u);
  EXPECT_EQ(copy.ResolveStructType(copy.GetType("output").ref_struct_type()).ref_type_id(), 2u);
  copy.Clear();
  EXPECT_TRUE(copy.Empty());
  EXPECT_EQ(copy.StructTypes().size(), 2u);
}

TEST(StructuredInference, NewModelsDoNotInheritDeclarations) {
  auto model = StructuredModel();
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  model.ref_struct_types().Clear();
  EXPECT_THROW(context.ComputeShapeModel(model), std::invalid_argument);
  EXPECT_TRUE(context.StructTypes().empty());
  EXPECT_FALSE(context.HasEncodedValue("output"));
}

TEST(StructuredInference, RejectsInvalidReferencesAndPayloads) {
  auto model = StructuredModel();
  core::shapes::ShapesContext context;
  context.SetStructTypes(model.ref_struct_types());
  TypeProto missing;
  missing.ref_struct_type().set_type_ref(uint64_t(42));
  EXPECT_THROW(context.SetType("bad", missing), std::invalid_argument);
  auto encoded = model.graph().ref_encoded_initializer()[0];
  encoded.set_raw_data("x");
  EXPECT_THROW(context.SetEncodedValue("bad", encoded), std::invalid_argument);
  EXPECT_FALSE(context.HasType("bad"));
  EXPECT_FALSE(context.HasEncodedValue("bad"));
  model.ref_struct_types()[1].ref_structure().ref_field()[0].ref_type() = missing;
  EXPECT_THROW(context.SetStructTypes(model.ref_struct_types()), std::invalid_argument);
  EXPECT_EQ(context.StructTypes().size(), 2u);
}

TEST(StructuredInference, StructuredInputsAndUnsupportedTransforms) {
  auto model = StructuredModel();
  auto &graph = model.ref_graph();
  graph.ref_encoded_initializer().Clear();
  auto *input = graph.add_input();
  input->set_name("encoded");
  input->ref_type().ref_struct_type().set_type_ref(uint64_t(2));
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  EXPECT_TRUE(context.HasType("output"));
  EXPECT_FALSE(context.Has("output"));
  EXPECT_THROW(context.ComputeShapeNode(UnaryNode("Abs", "output", "bad")), std::invalid_argument);
  model = StructuredModel();
  context.ComputeShapeModel(model);
  EXPECT_THROW(context.ComputeShapeNode(UnaryNode("Transpose", "output", "bad")),
               std::invalid_argument);
  context.Set("output", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
  EXPECT_FALSE(context.HasEncodedValue("output"));
  EXPECT_FALSE(context.HasType("output"));
}

TEST(StructuredInference, IfPreservesInheritedEncodingAndRejectsMismatch) {
  auto model = StructuredModel();
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  context.Set("condition",
              core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  auto left = Branch("left");
  auto right = Branch("right");
  context.ComputeShapeNode(IfNode(left, right));
  EXPECT_EQ(context.GetEncodedLayout("result").record_count, 2u);
  EXPECT_TRUE(context.GetType("result").has_struct_type());
  auto *different = right.ref_encoded_initializer().Add();
  *different = model.graph().ref_encoded_initializer()[0];
  different->set_name("different");
  different->set_raw_data(std::string(4, '\2'));
  right.ref_node()[0].ref_input()[0] = "different";
  EXPECT_THROW(context.ComputeShapeNode(IfNode(left, right)), std::invalid_argument);
}

TEST(StructuredInference, LoopPreservesCarriedEncoding) {
  auto model = StructuredModel();
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  context.Set("trip", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, {}));
  context.Set("condition",
              core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  GraphProto body;
  body.add_input()->set_name("iteration");
  body.add_input()->set_name("cond_in");
  body.add_input()->set_name("carried");
  *body.add_node() = UnaryNode("Identity", "cond_in", "cond_out");
  *body.add_node() = UnaryNode("Identity", "carried", "carried_out");
  body.add_output()->set_name("cond_out");
  body.add_output()->set_name("carried_out");
  NodeProto loop;
  loop.set_op_type("Loop");
  loop.add_input("trip");
  loop.add_input("condition");
  loop.add_input("encoded");
  loop.add_output("result");
  auto *attribute = loop.add_attribute();
  attribute->set_name("body");
  attribute->set_type(AttributeProto::GRAPH);
  attribute->set_g(body);
  context.ComputeShapeNode(loop);
  EXPECT_EQ(context.GetEncodedLayout("result").element_bits, 16u);
  EXPECT_EQ(context.Get("result").Shape()[0].AsInt(), 8);
}

TEST(StructuredInference, AffineIdentityKeepsCodesSeparateFromLogicalTensor) {
  EncodedValueProto encoded;
  encoded.ref_affine().set_storage_type(TensorProto::INT4);
  auto &scale = encoded.ref_affine().ref_scale();
  scale.set_data_type(TensorProto::FLOAT);
  scale.ref_float_data().push_back(0.5f);
  encoded.ref_logical_type() = LogicalTensor();
  encoded.set_raw_data(std::string(4, '\1'));
  core::shapes::ShapesContext context;
  context.SetEncodedValue("encoded", encoded);
  context.ComputeShapeNode(UnaryNode("Identity", "encoded", "copy"));
  auto layout = context.GetEncodedLayout("copy");
  EXPECT_EQ(layout.root, nullptr);
  ASSERT_NE(layout.affine, nullptr);
  EXPECT_EQ(layout.affine->ref_storage_type(), TensorProto::INT4);
  EXPECT_EQ(layout.element_bits, 4u);
  EXPECT_EQ(layout.payload_bytes, 4u);
  EXPECT_EQ(context.Get("copy").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(context.Get("copy").Data(), nullptr);
  EXPECT_FALSE(context.Get("copy").HasValueAsShape());
  encoded.ref_affine().set_storage_type(TensorProto::FLOAT);
  EXPECT_THROW(context.SetEncodedValue("bad", encoded), std::invalid_argument);
}

TEST(StructuredInference, LocalFunctionInheritsStructuredCatalogue) {
  auto model = StructuredModel();
  auto *function = model.add_functions();
  function->set_domain("custom");
  function->set_name("Forward");
  function->add_input("input");
  function->add_output("output");
  *function->add_node() = UnaryNode("Identity", "input", "output");
  auto &call = model.ref_graph().ref_node()[0];
  call.set_domain("custom");
  call.set_op_type("Forward");
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  EXPECT_EQ(context.GetEncodedLayout("output").element_bits, 16u);
}

TEST(StructuredInference, RejectsUnresolvedNestedInputType) {
  ModelProto model;
  auto *input = model.ref_graph().add_input();
  input->set_name("input");
  input->ref_type().ref_sequence_type().ref_elem_type().ref_struct_type().set_type_ref(uint64_t(9));
  core::shapes::ShapesContext context;
  EXPECT_THROW(context.ComputeShapeModel(model), std::invalid_argument);
}

TEST(StructuredInference, RejectsCrossModelTypeIdentityCollisions) {
  auto model = StructuredModel();
  core::shapes::ShapesContext original;
  original.ComputeShapeModel(model);
  core::shapes::ShapesContext same;
  same.SetStructTypes(model.ref_struct_types());
  EXPECT_NO_THROW(same.CopyValueFrom("copy", original, "encoded"));
  EXPECT_NO_THROW(original.CheckStructuredCompatibility("encoded", same, "copy"));

  model.ref_struct_types()[0].ref_array().ref_element_type().ref_tensor_type().set_elem_type(
      TensorProto::INT8);
  core::shapes::ShapesContext different;
  different.ComputeShapeModel(model);
  EXPECT_THROW(different.CopyValueFrom("copy", original, "encoded"), std::invalid_argument);
  EXPECT_FALSE(different.HasType("copy"));
  EXPECT_THROW(original.CheckStructuredCompatibility("encoded", different, "encoded"),
               std::invalid_argument);
}

TEST(StructuredInference, RegisteredCustomInferenceConsumesStructuredValues) {
  auto model = StructuredModel();
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  context.SetCustomShapeInferenceFunction(
      "custom", "StructuredForward", [](core::shapes::ShapesContext &ctx, const NodeProto &node) {
        ctx.CopyValueFrom(node.output(0), ctx, node.input(0));
      });
  auto node = UnaryNode("StructuredForward", "encoded", "custom_output");
  node.set_domain("custom");
  context.ComputeShapeNode(node);
  EXPECT_EQ(context.GetEncodedLayout("custom_output").element_bits, 16u);
  EXPECT_EQ(context.GetType("custom_output").ref_struct_type().ref_type_ref(), 2u);
}

TEST(StructuredInference, SubgraphStructuredInputsShadowCapturedValues) {
  auto model = StructuredModel();
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  context.Set("condition",
              core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  auto left = Branch("left");
  auto right = Branch("right");
  for (GraphProto *branch : {&left, &right}) {
    auto *input = branch->add_input();
    input->set_name("encoded");
    input->ref_type().ref_struct_type().set_type_ref(uint64_t(1));
  }
  context.ComputeShapeNode(IfNode(left, right));
  EXPECT_EQ(context.GetType("result").ref_struct_type().ref_type_ref(), 1u);
  EXPECT_FALSE(context.HasEncodedValue("result"));
  EXPECT_FALSE(context.Has("result"));
  EXPECT_EQ(context.GetType("encoded").ref_struct_type().ref_type_ref(), 2u);
  EXPECT_TRUE(context.HasEncodedValue("encoded"));
}

TEST(StructuredInference, EraseDropsAllValueDescriptorsAndPreservesScope) {
  auto model = StructuredModel();
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  context.SetOpsetVersion("", 24);
  context.Erase("output");
  EXPECT_FALSE(context.Has("output"));
  EXPECT_FALSE(context.HasType("output"));
  EXPECT_FALSE(context.HasEncodedValue("output"));
  EXPECT_TRUE(context.HasEncodedValue("encoded"));
  EXPECT_EQ(context.StructTypes().size(), 2u);
  EXPECT_EQ(context.OpsetVersion(""), 24);
  context.SetCustomShapeInferenceFunction("custom", "Forward",
                                          [](core::shapes::ShapesContext &, const NodeProto &) {});
  context.ClearValues();
  EXPECT_TRUE(context.Empty());
  EXPECT_FALSE(context.HasEncodedValue("encoded"));
  EXPECT_EQ(context.StructTypes().size(), 2u);
  EXPECT_EQ(context.OpsetVersion(""), 24);
  EXPECT_NE(context.GetCustomShapeInferenceFunction("custom", "Forward"), nullptr);
}

TEST(StructuredInference, ModelInferencePreservesOrdinaryPreseededDimensions) {
  ModelProto model;
  auto *input = model.ref_graph().add_input();
  input->set_name("input");
  input->ref_type() = LogicalTensor();
  *model.ref_graph().add_node() = UnaryNode("Identity", "input", "output");
  model.ref_graph().add_output()->set_name("output");
  core::shapes::ShapesContext context;
  context.Set("input", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                                 core::symbolic::SymShape{core::symbolic::SymDim(
                                                     std::string("custom_batch"))}));
  context.ComputeShapeModel(model);
  EXPECT_EQ(context.Get("input").Shape()[0].AsExpr(), "custom_batch");
  EXPECT_EQ(context.Get("output").Shape()[0].AsExpr(), "custom_batch");
}

TEST(StructuredInference, TensorTypesWithoutShapesKeepUnknownRank) {
  TypeProto type;
  type.ref_tensor_type().set_elem_type(TensorProto::FLOAT);
  core::shapes::ShapesContext context;
  context.SetType("input", type);
  EXPECT_TRUE(context.HasType("input"));
  EXPECT_FALSE(context.Has("input"));
  context.ComputeShapeNode(UnaryNode("Identity", "input", "output"));
  EXPECT_TRUE(context.HasType("output"));
  EXPECT_FALSE(context.Has("output"));
  EXPECT_FALSE(context.GetType("output").ref_tensor_type().has_shape());
}

TEST(StructuredInference, OrdinarySequenceOptionalAndMapInputsRemainSupported) {
  core::shapes::ShapesContext context;
  TypeProto sequence;
  sequence.ref_sequence_type().ref_elem_type() = LogicalTensor();
  context.SetType("sequence", sequence);
  ASSERT_TRUE(context.HasSequence("sequence"));
  EXPECT_FALSE(core::shapes::HasStructuredType(sequence));
  context.ComputeShapeNode(UnaryNode("SequenceLength", "sequence", "length"));
  EXPECT_EQ(context.Get("length").Dtype(), core::symbolic::TensorType::kInt64);
  context.ComputeShapeNode(UnaryNode("Identity", "sequence", "copied_sequence"));
  EXPECT_EQ(context.GetSequence("copied_sequence").Length(),
            context.GetSequence("sequence").Length());

  TypeProto optional;
  optional.ref_optional_type().ref_elem_type() = LogicalTensor();
  context.SetType("optional", optional);
  context.ComputeShapeNode(UnaryNode("OptionalGetElement", "optional", "element"));
  EXPECT_EQ(context.Get("element").Shape()[0].AsInt(), 8);

  TypeProto map;
  map.ref_map_type().set_key_type(TensorProto::INT64);
  map.ref_map_type().ref_value_type() = LogicalTensor();
  context.SetType("map", map);
  EXPECT_TRUE(context.Has("map"));
  EXPECT_FALSE(core::shapes::HasStructuredType(map));
}

TEST(StructuredInference, NestedStructuredContainersStillRequireCustomInference) {
  auto model = StructuredModel();
  core::shapes::ShapesContext context;
  context.SetStructTypes(model.ref_struct_types());
  TypeProto sequence;
  sequence.ref_sequence_type().ref_elem_type().ref_struct_type().set_type_ref(uint64_t(1));
  context.SetType("sequence", sequence);
  EXPECT_TRUE(core::shapes::HasStructuredType(sequence));
  EXPECT_THROW(context.ComputeShapeNode(UnaryNode("SequenceLength", "sequence", "length")),
               std::invalid_argument);
}

TEST(StructuredInference, IfAndLoopPreserveNestedStructuredContainers) {
  auto model = StructuredModel();
  TypeProto type;
  type.ref_optional_type()
      .ref_elem_type()
      .ref_sequence_type()
      .ref_elem_type()
      .ref_map_type()
      .set_key_type(TensorProto::INT64);
  type.ref_optional_type()
      .ref_elem_type()
      .ref_sequence_type()
      .ref_elem_type()
      .ref_map_type()
      .ref_value_type()
      .ref_struct_type()
      .set_type_ref(uint64_t(1));
  core::shapes::ShapesContext context;
  context.SetStructTypes(model.ref_struct_types());
  context.SetType("encoded", type);
  context.Set("condition",
              core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kBool, {}));
  context.ComputeShapeNode(IfNode(Branch("left"), Branch("right")));
  EXPECT_EQ(context.GetType("result").SerializeAsString(), type.SerializeAsString());
  EXPECT_FALSE(context.Has("result"));
  EXPECT_FALSE(context.HasEncodedValue("result"));

  GraphProto body;
  body.add_input()->set_name("iteration");
  body.add_input()->set_name("cond_in");
  body.add_input()->set_name("carried");
  *body.add_node() = UnaryNode("Identity", "cond_in", "cond_out");
  *body.add_node() = UnaryNode("Identity", "carried", "carried_out");
  body.add_output()->set_name("cond_out");
  body.add_output()->set_name("carried_out");
  NodeProto loop;
  loop.set_op_type("Loop");
  loop.add_input("");
  loop.add_input("condition");
  loop.add_input("result");
  loop.add_output("loop_result");
  auto *attribute = loop.add_attribute();
  attribute->set_name("body");
  attribute->set_type(AttributeProto::GRAPH);
  attribute->set_g(body);
  context.ComputeShapeNode(loop);
  EXPECT_EQ(context.GetType("loop_result").SerializeAsString(), type.SerializeAsString());
  EXPECT_FALSE(context.Has("loop_result"));
  EXPECT_FALSE(context.HasEncodedValue("loop_result"));

  auto different = Branch("different");
  auto *input = different.add_input();
  input->set_name("encoded");
  input->ref_type().ref_sequence_type().ref_elem_type().ref_struct_type().set_type_ref(uint64_t(1));
  EXPECT_THROW(context.ComputeShapeNode(IfNode(Branch("left"), different)), std::invalid_argument);
}

TEST(StructuredInference, RejectsIncompatibleNestedStructuredOutputDeclaration) {
  auto model = StructuredModel();
  auto &graph = model.ref_graph();
  graph.ref_encoded_initializer().Clear();
  auto *input = graph.add_input();
  input->set_name("encoded");
  input->ref_type().ref_sequence_type().ref_elem_type().ref_struct_type().set_type_ref(uint64_t(1));
  graph.ref_output()[0]
      .ref_type()
      .ref_sequence_type()
      .ref_elem_type()
      .ref_struct_type()
      .set_type_ref(uint64_t(2));
  core::shapes::ShapesContext context;
  EXPECT_THROW(context.ComputeShapeModel(model), std::invalid_argument);
}

TEST(StructuredInference, EncodedDefaultsPreservePublicTensorShapeAndValidatePayload) {
  auto model = StructuredModel();
  auto *input = model.ref_graph().add_input();
  input->set_name("encoded");
  input->ref_type() = LogicalTensor();
  input->ref_type().ref_tensor_type().ref_shape().ref_dim()[0].set_dim_param("batch");
  model.ref_graph().ref_node()[0].set_op_type("Abs");
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  EXPECT_FALSE(context.HasEncodedValue("encoded"));
  EXPECT_FALSE(context.HasEncodedValue("output"));
  EXPECT_EQ(context.Get("output").Shape()[0].AsExpr(), "batch");

  core::shapes::ShapesContext preseeded;
  preseeded.Set("encoded",
                core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                          core::symbolic::SymShape{core::symbolic::SymDim(12)}));
  preseeded.ComputeShapeModel(model);
  EXPECT_EQ(preseeded.Get("output").Shape()[0].AsInt(), 12);

  model.ref_graph().ref_encoded_initializer()[0].set_raw_data("x");
  EXPECT_THROW(context.ComputeShapeModel(model), std::invalid_argument);
  model.ref_graph().ref_encoded_initializer()[0].set_raw_data(std::string(4, '\1'));
  input->ref_type().ref_tensor_type().set_elem_type(TensorProto::INT64);
  EXPECT_THROW(context.ComputeShapeModel(model), std::invalid_argument);
}

TEST(StructuredInference, ReorderedCataloguesRemainCompatibleForCopyAndControlFlow) {
  auto model = StructuredModel();
  core::shapes::ShapesContext original;
  original.ComputeShapeModel(model);
  std::swap(model.ref_struct_types()[0], model.ref_struct_types()[1]);
  core::shapes::ShapesContext reordered;
  reordered.SetStructTypes(model.ref_struct_types());
  EXPECT_NO_THROW(reordered.CopyValueFrom("copy", original, "encoded"));
  EXPECT_NO_THROW(original.CheckStructuredCompatibility("encoded", reordered, "copy"));
  EXPECT_EQ(reordered.GetEncodedLayout("copy").element_bits, 16u);
  EXPECT_EQ(reordered.ResolveStructType(reordered.GetType("copy").ref_struct_type()).ref_type_id(),
            2u);
  StructTypeProto unrelated = model.ref_struct_types()[1];
  unrelated.set_type_id(uint64_t(3));
  model.ref_struct_types().push_back(std::move(unrelated));
  core::shapes::ShapesContext extended;
  extended.SetStructTypes(model.ref_struct_types());
  EXPECT_NO_THROW(extended.CopyValueFrom("copy", original, "encoded"));
  EXPECT_NO_THROW(original.CheckStructuredCompatibility("encoded", extended, "copy"));
}

} // namespace Test
