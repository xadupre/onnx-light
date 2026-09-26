#include "onnx_verify.h"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto MakeValidModel() {
  ModelProto model;
  model.add_opset("", 18);
  GraphProto &graph = *model.add_graph();
  graph.set_name("g");

  ValueInfoProto *input = graph.add_input();
  input->set_name("x");
  input->ref_type().ref_tensor_type().set_elem_type(TensorProto::FLOAT);
  input->ref_type().ref_tensor_type().ref_shape().ref_dim();

  ValueInfoProto *output = graph.add_output();
  output->set_name("y");
  output->ref_type().ref_tensor_type().set_elem_type(TensorProto::FLOAT);
  output->ref_type().ref_tensor_type().ref_shape().ref_dim();

  NodeProto *node = graph.add_node();
  node->set_op_type("Identity");
  node->add_input("x");
  node->add_output("y");
  return model;
}

ModelProto MakePersistentModel() {
  ModelProto model = MakeValidModel();
  auto *binding = model.mutable_graph()->add_persistent_bindings();
  binding->set_input_name("x");
  binding->set_output_name("y");
  return model;
}

GraphProto MakeCapturingGraph() {
  GraphProto graph = MakeValidModel().graph();
  graph.clear_input();
  return graph;
}

GraphProto *AddNestedGraph(NodeProto &node, bool repeated) {
  auto *attribute = node.add_attribute();
  attribute->set_name(repeated ? "bodies" : "body");
  attribute->set_type(repeated ? AttributeProto::GRAPHS : AttributeProto::GRAPH);
  return repeated ? attribute->add_graphs() : attribute->mutable_g();
}

void ExpectPersistentBindingError(const GraphProto &graph, const std::string &message) {
  try {
    VerifyPersistentBindings(nullptr, graph);
    FAIL() << "Invalid persistent binding was accepted.";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
  }
}

} // namespace

TEST(onnx_verify, VerifyModel_Valid) {
  ModelProto model = MakeValidModel();
  EXPECT_NO_THROW(VerifyModel(model));
}

TEST(onnx_verify, PagedKVCacheTypeIsNamedAndVersioned) {
  const auto type = PagedKVCacheTypeV1();
  ASSERT_TRUE(type.has_struct_type());
  EXPECT_EQ(type.struct_type().name(), "onnx_light.PagedKVCache");
  ASSERT_EQ(type.struct_type().metadata_props().size(), 1u);
  EXPECT_EQ(type.struct_type().metadata_props(0).key(), "onnx_light.type_version");
  EXPECT_EQ(type.struct_type().metadata_props(0).value(), "1");
  StructTypeCatalogue catalogue;
  EXPECT_NO_THROW(ValidatePersistentType(catalogue, type));
}

TEST(onnx_verify, PagedCacheRejectsEncodedLogicalDimensionsWithSymbolicAlternatives) {
  ModelProto model = MakeValidModel();
  auto *cache = model.mutable_graph()->add_paged_cache_initializer();
  cache->set_name("cache");
  auto *block = cache->add_blocks();
  block->set_start(0);
  block->set_length(1);
  auto *encoded = block->mutable_encoded_key();
  encoded->mutable_affine()->set_storage_type(TensorProto::INT8);
  auto *scale = encoded->mutable_affine()->mutable_scale();
  scale->set_data_type(TensorProto::FLOAT);
  scale->add_float_data(1.f);
  auto *tensor = encoded->mutable_logical_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  for (int64_t dim : {1, 1, 2, 2})
    tensor->mutable_shape()->add_dim()->set_dim_value(dim);
  encoded->set_raw_data(std::string(4, '\0'));
  auto *value = block->mutable_value();
  value->set_data_type(TensorProto::FLOAT);
  for (int64_t dim : {1, 1, 2, 2})
    value->add_dims(dim);
  for (float element : {1.f, 2.f, 3.f, 4.f})
    value->add_float_data(element);
  EXPECT_NO_THROW(VerifyModel(model));
  tensor->mutable_shape()->mutable_dim(0)->set_dim_param("N");
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
}

TEST(onnx_verify, PersistentBindings_RoundtripAndRootOnly) {
  ModelProto model = MakeValidModel();
  auto *binding = model.mutable_graph()->add_persistent_bindings();
  binding->set_input_name("x");
  binding->set_output_name("y");
  EXPECT_NO_THROW(VerifyModel(model));
  ModelProto restored;
  ASSERT_TRUE(restored.ParseFromString(model.SerializeAsString()));
  ASSERT_EQ(restored.graph().persistent_bindings_size(), 1);
  EXPECT_EQ(restored.graph().persistent_bindings(0).input_name(), "x");
  EXPECT_NO_THROW(VerifyModel(restored));
  EXPECT_THROW(VerifyPersistentBindings(nullptr, restored.graph(), false), std::invalid_argument);
  restored.mutable_graph()->add_persistent_bindings(*binding);
  EXPECT_THROW(VerifyModel(restored), std::invalid_argument);
}

TEST(onnx_verify, PersistentBindings_FixedShapesAndNames) {
  ModelProto model = MakeValidModel();
  auto *binding = model.mutable_graph()->add_persistent_bindings();
  binding->set_input_name("missing");
  binding->set_output_name("y");
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
  binding->set_input_name("x");
  model.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::INT32);
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
  model.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::FLOAT);
  model.mutable_graph()
      ->mutable_input(0)
      ->mutable_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->add_dim()
      ->set_dim_param("N");
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
}

TEST(onnx_verify, PersistentBindings_WholeStructureCompatibility) {
  TypeProto type;
  auto *structure = type.mutable_struct_type()->mutable_structure();
  auto *field = structure->add_field();
  field->set_name("a.b");
  auto *tensor = field->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape()->add_dim()->set_dim_value(2);
  StructTypeCatalogue catalogue;
  EXPECT_TRUE(CompatiblePersistentTypes(catalogue, type, type));
  EXPECT_TRUE(CompatiblePersistentStructTypes(catalogue, type.struct_type(), type.struct_type()));
  TypeProto different = type;
  different.mutable_struct_type()
      ->mutable_structure()
      ->mutable_field(0)
      ->mutable_type()
      ->mutable_tensor_type()
      ->set_elem_type(TensorProto::INT32);
  EXPECT_FALSE(CompatiblePersistentTypes(catalogue, type, different));
  EXPECT_FALSE(
      CompatiblePersistentStructTypes(catalogue, type.struct_type(), different.struct_type()));
}

TEST(onnx_verify, PersistentBindings_RejectsStringsInEitherEndpoint) {
  for (bool input : {false, true}) {
    ModelProto model = MakeValidModel();
    auto *value =
        input ? model.mutable_graph()->mutable_input(0) : model.mutable_graph()->mutable_output(0);
    value->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::STRING);
    auto *binding = model.mutable_graph()->add_persistent_bindings();
    binding->set_input_name("x");
    binding->set_output_name("y");
    try {
      VerifyPersistentBindings(nullptr, model.graph());
      FAIL() << "String endpoint was accepted.";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find("String tensors cannot be persistent"),
                std::string::npos);
    }
    model.mutable_graph()->clear_persistent_bindings();
    EXPECT_NO_THROW(VerifyModel(model));
  }
}

TEST(onnx_verify, PersistentTypes_TraversesContainerFieldsAndCatalogueDiamonds) {
  StructTypeCatalogue catalogue;
  TypeProto strings;
  strings.mutable_tensor_type()->set_elem_type(TensorProto::STRING);
  TypeProto sequence, optional, map, array, sparse;
  *sequence.mutable_sequence_type()->mutable_elem_type() = strings;
  *optional.mutable_optional_type()->mutable_elem_type() = strings;
  map.mutable_map_type()->set_key_type(TensorProto::INT64);
  *map.mutable_map_type()->mutable_value_type() = strings;
  array.mutable_struct_type()->mutable_array()->set_dimension(2);
  *array.mutable_struct_type()->mutable_array()->mutable_element_type() = strings;
  sparse.mutable_sparse_tensor_type()->set_elem_type(TensorProto::STRING);
  for (const auto &type : {strings, sequence, optional, map, array, sparse}) {
    EXPECT_NO_THROW(catalogue.ValidateType(type));
    EXPECT_THROW(ValidatePersistentType(catalogue, type), std::invalid_argument);
  }

  ModelProto model;
  auto *leaf = model.add_struct_types();
  leaf->set_type_id(1);
  auto *field = leaf->mutable_structure()->add_field();
  field->set_name("value");
  field->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  for (uint64_t id = 2; id < 20; ++id) {
    auto *parent = model.add_struct_types();
    parent->set_type_id(id);
    for (const char *name : {"left", "right"}) {
      auto *child = parent->mutable_structure()->add_field();
      child->set_name(name);
      child->mutable_type()->mutable_struct_type()->set_type_ref(id - 1);
    }
  }
  catalogue.Build(model);
  TypeProto root;
  root.mutable_struct_type()->set_type_ref(19);
  EXPECT_NO_THROW(ValidatePersistentType(catalogue, root));
  model.mutable_struct_types(0)
      ->mutable_structure()
      ->mutable_field(0)
      ->mutable_type()
      ->mutable_tensor_type()
      ->set_elem_type(TensorProto::STRING);
  catalogue.Build(model);
  EXPECT_THROW(ValidatePersistentType(catalogue, root), std::invalid_argument);
  StructTypeCatalogue empty;
  EXPECT_THROW(ValidatePersistentType(empty, root), std::invalid_argument);
}

TEST(onnx_verify, PersistentTypes_SequenceCompatibilityAndUnsupportedElements) {
  StructTypeCatalogue catalogue;
  TypeProto sequence;
  auto *tensor = sequence.mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape()->add_dim()->set_dim_value(2);
  EXPECT_NO_THROW(ValidatePersistentType(catalogue, sequence));
  EXPECT_TRUE(CompatiblePersistentTypes(catalogue, sequence, sequence));
  TypeProto other = sequence;
  other.mutable_sequence_type()
      ->mutable_elem_type()
      ->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_dim(0)
      ->set_dim_value(3);
  EXPECT_FALSE(CompatiblePersistentTypes(catalogue, sequence, other));
  other = sequence;
  other.mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::INT64);
  EXPECT_FALSE(CompatiblePersistentTypes(catalogue, sequence, other));
  EXPECT_FALSE(
      CompatiblePersistentTypes(catalogue, sequence, sequence.sequence_type().elem_type()));
  TypeProto optional, map, sparse, opaque, unset;
  *optional.mutable_optional_type()->mutable_elem_type() = sequence;
  map.mutable_map_type()->set_key_type(TensorProto::INT64);
  *map.mutable_map_type()->mutable_value_type() = sequence;
  sparse.mutable_sparse_tensor_type()->set_elem_type(TensorProto::FLOAT);
  opaque.mutable_opaque_type();
  for (const auto &type : {optional, map, sparse, opaque, unset}) {
    EXPECT_THROW(ValidatePersistentType(catalogue, type), std::invalid_argument);
    TypeProto nested;
    *nested.mutable_sequence_type()->mutable_elem_type() = type;
    EXPECT_THROW(ValidatePersistentType(catalogue, nested), std::invalid_argument);
  }
  TypeProto nested;
  auto *element = &nested;
  for (size_t i = 0; i < 65; ++i)
    element = element->mutable_sequence_type()->mutable_elem_type();
  element->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  EXPECT_THROW(ValidatePersistentType(catalogue, nested), std::invalid_argument);
}

TEST(onnx_verify, PersistentBindings_RejectsLegacyFieldPathWire) {
  for (const auto &wire : {std::string("\x1a\x05"
                                       "cache"),
                           std::string("\x22\x05"
                                       "cache")}) {
    PersistentBindingProto binding;
    EXPECT_FALSE(binding.ParseFromString(wire));
  }
}

TEST(onnx_verify, PersistentBindings_ExactNamesAndDuplicateSources) {
  ModelProto model = MakeValidModel();
  model.mutable_graph()->mutable_input(0)->set_name("state.in");
  model.mutable_graph()->mutable_output(0)->set_name("state.out");
  model.mutable_graph()->mutable_node(0)->ref_input()[0] = "state.in";
  model.mutable_graph()->mutable_node(0)->ref_output()[0] = "state.out";
  auto *binding = model.mutable_graph()->add_persistent_bindings();
  binding->set_input_name("state.in");
  binding->set_output_name("state.out");
  EXPECT_NO_THROW(VerifyModel(model));
  binding->set_input_name("state.in.cache");
  ExpectPersistentBindingError(model.graph(), "exact existing typed graph input/output name");
  binding->set_input_name("state.in");
  binding->set_output_name("state.out.cache");
  ExpectPersistentBindingError(model.graph(), "exact existing typed graph input/output name");
  binding->set_output_name("state.out");
  auto *input = model.mutable_graph()->add_input();
  input->set_name("other");
  *input->mutable_type() = model.graph().input(0).type();
  auto *second = model.mutable_graph()->add_persistent_bindings();
  second->set_input_name("other");
  second->set_output_name("state.out");
  ExpectPersistentBindingError(model.graph(), "duplicate output names");
}

TEST(onnx_verify, PersistentBindings_RejectsUnusedInput) {
  ModelProto model = MakePersistentModel();
  auto &graph = *model.mutable_graph();
  auto *ordinary = graph.add_input();
  ordinary->set_name("ordinary");
  *ordinary->mutable_type() = graph.input(0).type();
  graph.mutable_node(0)->ref_input()[0] = "ordinary";
  *graph.add_value_info() = graph.input(0);
  ExpectPersistentBindingError(
      graph, "Persistent input 'x' must be used exactly once; found 0 value-uses");
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
  graph.clear_persistent_bindings();
  EXPECT_NO_THROW(VerifyModel(model));
}

TEST(onnx_verify, PersistentBindings_IgnoresMetadataAndEmptyOptionalInputs) {
  ModelProto model = MakePersistentModel();
  auto &graph = *model.mutable_graph();
  *graph.add_value_info() = graph.input(0);
  *graph.add_value_info() = graph.output(0);
  graph.mutable_node(0)->add_input("");
  EXPECT_NO_THROW(VerifyModel(model));
}

TEST(onnx_verify, PersistentBindings_RejectsMultipleConsumersIncludingReadOnly) {
  for (const char *consumer : {"Identity", "Shape"}) {
    SCOPED_TRACE(consumer);
    ModelProto model = MakePersistentModel();
    auto &graph = *model.mutable_graph();
    graph.mutable_node(0)->set_op_type("Attention");
    auto *second = graph.add_node();
    second->set_op_type(consumer);
    second->add_input("x");
    second->add_output("observed");
    ExpectPersistentBindingError(
        graph, "Persistent input 'x' must be used exactly once; found 2 value-uses");
    EXPECT_THROW(VerifyModel(model), std::invalid_argument);
    graph.clear_persistent_bindings();
    EXPECT_NO_THROW(VerifyModel(model));
  }
}

TEST(onnx_verify, PersistentBindings_CountsEveryInputPosition) {
  ModelProto model = MakePersistentModel();
  auto *node = model.mutable_graph()->mutable_node(0);
  node->set_op_type("Add");
  node->add_input("x");
  ExpectPersistentBindingError(
      model.graph(), "Persistent input 'x' must be used exactly once; found 2 value-uses");
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
  node->add_input("x");
  ExpectPersistentBindingError(
      model.graph(), "Persistent input 'x' must be used exactly once; found 3 value-uses");
}

TEST(onnx_verify, PersistentBindings_ChecksEveryBoundInput) {
  ModelProto model = MakePersistentModel();
  auto &graph = *model.mutable_graph();
  auto *input = graph.add_input();
  *input = graph.input(0);
  input->set_name("other");
  auto *output = graph.add_output();
  *output = graph.output(0);
  output->set_name("next");
  auto *node = graph.add_node();
  node->set_op_type("Identity");
  node->add_input("other");
  node->add_output("next");
  auto *binding = graph.add_persistent_bindings();
  binding->set_input_name("other");
  binding->set_output_name("next");
  EXPECT_NO_THROW(VerifyModel(model));

  node->add_input("other");
  ExpectPersistentBindingError(
      graph, "Persistent input 'other' must be used exactly once; found 2 value-uses");
  node->clear_input();
  ExpectPersistentBindingError(
      graph, "Persistent input 'other' must be used exactly once; found 0 value-uses");
}

TEST(onnx_verify, PersistentBindings_CountsDirectGraphOutput) {
  ModelProto model = MakePersistentModel();
  auto &graph = *model.mutable_graph();
  graph.clear_node();
  graph.mutable_output(0)->set_name("x");
  graph.mutable_persistent_bindings(0)->set_output_name("x");
  EXPECT_NO_THROW(VerifyModel(model));

  auto *node = graph.add_node();
  node->set_op_type("Identity");
  node->add_input("x");
  node->add_output("y");
  ExpectPersistentBindingError(
      graph, "Persistent input 'x' must be used exactly once; found 2 value-uses");
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
  graph.clear_persistent_bindings();
  EXPECT_NO_THROW(VerifyModel(model));
}

TEST(onnx_verify, PersistentBindings_CountsNestedGraphAndGraphsCaptures) {
  for (bool outer_repeated : {false, true}) {
    for (bool inner_repeated : {false, true}) {
      SCOPED_TRACE(outer_repeated);
      SCOPED_TRACE(inner_repeated);
      ModelProto model = MakePersistentModel();
      auto *node = model.mutable_graph()->mutable_node(0);
      node->set_op_type("WithSubgraph");
      node->clear_input();
      auto *outer = AddNestedGraph(*node, outer_repeated);
      *outer = MakeCapturingGraph();
      *outer->add_value_info() = model.graph().input(0);
      auto *nested = outer->mutable_node(0);
      nested->set_op_type("WithSubgraph");
      nested->clear_input();
      auto *inner = AddNestedGraph(*nested, inner_repeated);
      *inner = MakeCapturingGraph();
      *inner->add_value_info() = model.graph().input(0);
      EXPECT_NO_THROW(VerifyModel(model));

      node->add_input("x");
      ExpectPersistentBindingError(
          model.graph(), "Persistent input 'x' must be used exactly once; found 2 value-uses");
      EXPECT_THROW(VerifyModel(model), std::invalid_argument);
      inner->mutable_node(0)->add_input("x");
      ExpectPersistentBindingError(
          model.graph(), "Persistent input 'x' must be used exactly once; found 3 value-uses");
    }
  }
}

TEST(onnx_verify, PersistentBindings_CountsEveryBranchOutputCapture) {
  for (bool repeated : {false, true}) {
    SCOPED_TRACE(repeated);
    ModelProto model = MakePersistentModel();
    auto *node = model.mutable_graph()->mutable_node(0);
    node->clear_input();
    node->set_op_type(repeated ? "WithSubgraphs" : "If");
    if (!repeated) {
      auto *condition = model.mutable_graph()->add_input();
      condition->set_name("condition");
      auto *type = condition->mutable_type()->mutable_tensor_type();
      type->set_elem_type(TensorProto::BOOL);
      type->mutable_shape()->ref_dim();
      node->add_input("condition");
    }
    auto *first = AddNestedGraph(*node, repeated);
    first->set_name("then");
    first->add_output()->set_name("x");
    EXPECT_NO_THROW(VerifyModel(model));

    GraphProto *second;
    if (repeated) {
      second = node->mutable_attribute(0)->add_graphs();
    } else {
      node->mutable_attribute(0)->set_name("then_branch");
      second = AddNestedGraph(*node, false);
      node->mutable_attribute(1)->set_name("else_branch");
    }
    second->set_name("else");
    second->add_output()->set_name("x");
    ExpectPersistentBindingError(
        model.graph(), "Persistent input 'x' must be used exactly once; found 2 value-uses");
    EXPECT_THROW(VerifyModel(model), std::invalid_argument);
    model.mutable_graph()->clear_persistent_bindings();
    EXPECT_NO_THROW(VerifyModel(model));
  }
}

TEST(onnx_verify, PersistentBindings_RespectsLocalDefinitionsAndRestoresSiblingScope) {
  for (int definition = 0; definition < 5; ++definition) {
    SCOPED_TRACE(definition);
    ModelProto model = MakePersistentModel();
    auto *node = model.mutable_graph()->mutable_node(0);
    auto *outer = AddNestedGraph(*node, false);
    outer->set_name("shadowing");
    switch (definition) {
    case 0:
      outer->add_input()->set_name("x");
      break;
    case 1:
      outer->add_initializer()->set_name("x");
      break;
    case 2:
      outer->add_sparse_initializer()->mutable_values()->set_name("x");
      break;
    case 3:
      outer->add_encoded_initializer()->set_name("x");
      break;
    case 4:
      auto *producer = outer->add_node();
      producer->set_op_type("Constant");
      producer->add_output("x");
      break;
    }
    outer->add_output()->set_name("x");
    auto *nested = outer->add_node();
    nested->set_op_type("WithSubgraph");
    nested->add_output("nested");
    auto *inner = AddNestedGraph(*nested, false);
    *inner = MakeCapturingGraph();
    inner->add_input()->set_name("x");
    *AddNestedGraph(*nested, true) = MakeCapturingGraph();
    // Tests the binding counter alone; VerifyGraph separately enforces SSA and tensor validity.
    EXPECT_NO_THROW(VerifyPersistentBindings(nullptr, model.graph()));
    auto *sibling = AddNestedGraph(*node, true);
    *sibling = MakeCapturingGraph();
    ExpectPersistentBindingError(
        model.graph(), "Persistent input 'x' must be used exactly once; found 2 value-uses");
  }
}

TEST(onnx_verify, PersistentBindings_CountsFunctionCallArgumentsWithoutExpandingBodies) {
  ModelProto model = MakePersistentModel();
  model.add_opset("custom", 1);
  auto *function = model.add_functions();
  function->set_name("UseTwice");
  function->set_domain("custom");
  function->add_input("state");
  function->add_output("result");
  auto *body = function->add_node();
  body->set_op_type("Add");
  body->add_input("state");
  body->add_input("state");
  body->add_output("result");
  auto *call = model.mutable_graph()->mutable_node(0);
  call->set_op_type("UseTwice");
  call->set_domain("custom");
  EXPECT_NO_THROW(VerifyModel(model));

  function->add_input("other");
  call->add_input("x");
  ExpectPersistentBindingError(
      model.graph(), "Persistent input 'x' must be used exactly once; found 2 value-uses");
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
}

TEST(onnx_verify, PersistentBindings_TypeErrorsPrecedeUseCountErrors) {
  ModelProto model = MakePersistentModel();
  model.mutable_graph()->mutable_node(0)->clear_input();
  model.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::INT32);
  ExpectPersistentBindingError(model.graph(), "compatible tensor/struct/sequence types");
  model.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
      TensorProto::STRING);
  ExpectPersistentBindingError(model.graph(), "String tensors cannot be persistent");
}

TEST(onnx_verify, PersistentInputUses_DoesNotRequireTypeCatalogue) {
  ModelProto model = MakePersistentModel();
  auto &graph = *model.mutable_graph();
  TypeProto type;
  type.mutable_struct_type()->set_type_ref(123);
  *graph.mutable_input(0)->mutable_type() = type;
  *graph.mutable_output(0)->mutable_type() = type;
  EXPECT_NO_THROW(VerifyPersistentInputUses(graph));
  EXPECT_THROW(VerifyPersistentBindings(nullptr, graph), std::invalid_argument);

  graph.mutable_node(0)->add_input("x");
  try {
    VerifyPersistentInputUses(graph);
    FAIL() << "Repeated persistent input was accepted.";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::string(error.what())
                  .find("Persistent input 'x' must be used exactly once; found 2 value-uses"),
              std::string::npos);
  }
  graph.mutable_node(0)->clear_input();
  EXPECT_THROW(VerifyPersistentInputUses(graph), std::invalid_argument);
}

TEST(onnx_verify, PersistentInputUses_DoesNotValidateDeclarations) {
  ModelProto model = MakePersistentModel();
  auto &graph = *model.mutable_graph();
  graph.clear_input();
  graph.clear_output();
  const auto binding = graph.persistent_bindings(0);
  graph.add_persistent_bindings(binding);
  EXPECT_NO_THROW(VerifyPersistentInputUses(graph));
  EXPECT_THROW(VerifyPersistentBindings(nullptr, graph), std::invalid_argument);

  graph.clear_persistent_bindings();
  graph.mutable_node(0)->add_input("x");
  EXPECT_NO_THROW(VerifyPersistentInputUses(graph));
}

TEST(onnx_verify, PersistentBindings_PartialTensorDeclarations) {
  StructTypeCatalogue catalogue;
  TypeProto unknown, symbolic, concrete;
  unknown.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  symbolic.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  symbolic.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("N");
  concrete.mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  concrete.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  EXPECT_TRUE(CompatiblePersistentTypes(catalogue, unknown, concrete));
  EXPECT_TRUE(CompatiblePersistentTypes(catalogue, concrete, unknown));
  EXPECT_TRUE(CompatiblePersistentTypes(catalogue, symbolic, concrete));
  EXPECT_TRUE(CompatiblePersistentTypes(catalogue, concrete, symbolic));
  symbolic.mutable_tensor_type()->mutable_shape()->mutable_dim(0)->clear_dim_param();
  EXPECT_TRUE(CompatiblePersistentTypes(catalogue, symbolic, concrete));
  symbolic.mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_value(4);
  EXPECT_FALSE(CompatiblePersistentTypes(catalogue, symbolic, concrete));
  symbolic.mutable_tensor_type()->clear_shape();
  symbolic.mutable_tensor_type()->set_elem_type(TensorProto::INT32);
  EXPECT_FALSE(CompatiblePersistentTypes(catalogue, symbolic, concrete));

  ModelProto model = MakeValidModel();
  *model.mutable_graph()->mutable_input(0)->mutable_type() = unknown;
  *model.mutable_graph()->mutable_output(0)->mutable_type() = concrete;
  auto *binding = model.mutable_graph()->add_persistent_bindings();
  binding->set_input_name("x");
  binding->set_output_name("y");
  EXPECT_NO_THROW(VerifyPersistentBindings(&catalogue, model.graph()));
}

TEST(onnx_verify, VerifyModel_MissingGraph) {
  ModelProto model;
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
}

TEST(onnx_verify, VerifyModel_MissingOpset) {
  ModelProto model = MakeValidModel();
  model.ref_opset_import().clear();
  EXPECT_THROW(VerifyModel(model), std::invalid_argument);
}

TEST(onnx_verify, VerifyGraph_NotTopologicallySorted) {
  ModelProto model = MakeValidModel();
  GraphProto &graph = model.ref_graph();
  NodeProto *bad = graph.add_node();
  bad->set_op_type("Identity");
  bad->add_input("not_yet_defined");
  bad->add_output("z");
  EXPECT_THROW(VerifyGraph(graph), std::invalid_argument);
}

TEST(onnx_verify, VerifyGraph_DuplicateOutput_SSA) {
  ModelProto model = MakeValidModel();
  GraphProto &graph = model.ref_graph();
  NodeProto *node = graph.add_node();
  node->set_op_type("Identity");
  node->add_input("x");
  node->add_output("y"); // "y" already produced by the first node.
  EXPECT_THROW(VerifyGraph(graph), std::invalid_argument);
}

TEST(onnx_verify, VerifyGraph_UnproducedOutput) {
  ModelProto model = MakeValidModel();
  GraphProto &graph = model.ref_graph();
  ValueInfoProto *extra_output = graph.add_output();
  extra_output->set_name("never_produced");
  extra_output->ref_type().ref_tensor_type().set_elem_type(TensorProto::FLOAT);
  extra_output->ref_type().ref_tensor_type().ref_shape().ref_dim();
  EXPECT_THROW(VerifyGraph(graph), std::invalid_argument);
}

TEST(onnx_verify, VerifyNode_EmptyOpType) {
  NodeProto node;
  node.add_input("x");
  node.add_output("y");
  EXPECT_THROW(VerifyNode(node, false, {}), std::invalid_argument);
}

TEST(onnx_verify, VerifyNode_NoInputNoOutput) {
  NodeProto node;
  node.set_op_type("Identity");
  EXPECT_THROW(VerifyNode(node, false, {}), std::invalid_argument);
}

TEST(onnx_verify, VerifyNode_DuplicateAttribute) {
  NodeProto node;
  node.set_op_type("Cast");
  node.add_input("x");
  node.add_output("y");
  AttributeProto *a1 = node.add_attribute();
  a1->set_name("to");
  a1->set_type(AttributeProto::INT);
  a1->set_i(1);
  AttributeProto *a2 = node.add_attribute();
  a2->set_name("to");
  a2->set_type(AttributeProto::INT);
  a2->set_i(2);
  EXPECT_THROW(VerifyNode(node, false, {"x"}), std::invalid_argument);
}

TEST(onnx_verify, VerifyTensor_Valid) {
  TensorProto tensor;
  tensor.set_name("w");
  tensor.set_data_type(TensorProto::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_raw_data().push_back(0);
  tensor.ref_raw_data().push_back(0);
  tensor.ref_raw_data().push_back(0);
  tensor.ref_raw_data().push_back(0);
  tensor.ref_raw_data().push_back(0);
  tensor.ref_raw_data().push_back(0);
  tensor.ref_raw_data().push_back(0);
  tensor.ref_raw_data().push_back(0);
  EXPECT_NO_THROW(VerifyTensor(tensor));
}

TEST(onnx_verify, VerifyTensor_UndefinedDataType) {
  TensorProto tensor;
  tensor.set_name("w");
  EXPECT_THROW(VerifyTensor(tensor), std::invalid_argument);
}

TEST(onnx_verify, VerifyTensor_ZeroElementsWithData) {
  TensorProto tensor;
  tensor.set_name("w");
  tensor.set_data_type(TensorProto::FLOAT);
  tensor.ref_dims().push_back(0);
  tensor.ref_raw_data().push_back(1);
  EXPECT_THROW(VerifyTensor(tensor), std::invalid_argument);
}

// Non-packed int32_data-stored types require one int32 entry per element.
TEST(onnx_verify, VerifyTensor_UnpackedInt32DataTooSmall) {
  for (TensorProto::DataType dtype :
       {TensorProto::BOOL, TensorProto::INT8, TensorProto::UINT8, TensorProto::INT16,
        TensorProto::UINT16, TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT8E4M3FN,
        TensorProto::FLOAT8E4M3FNUZ, TensorProto::FLOAT8E5M2, TensorProto::FLOAT8E5M2FNUZ,
        TensorProto::FLOAT8E8M0}) {
    // 4 int32_data entries for a 32-element tensor: the 4-bit packed formula
    // ceil(32/8) = 4 would accept this, but these types need nelem=32 entries.
    TensorProto t;
    t.set_name("w");
    t.set_data_type(dtype);
    t.add_dims(32);
    for (int i = 0; i < 4; ++i) {
      t.add_int32_data(0);
    }
    EXPECT_THROW(VerifyTensor(t), std::invalid_argument);

    TensorProto ok;
    ok.set_name("w");
    ok.set_data_type(dtype);
    ok.add_dims(32);
    for (int i = 0; i < 32; ++i) {
      ok.add_int32_data(0);
    }
    EXPECT_NO_THROW(VerifyTensor(ok));
  }
}

// Packed 4-bit types need ceil(nelem/8) int32 entries.
TEST(onnx_verify, VerifyTensor_Packed4BitInt32DataTooSmall) {
  for (TensorProto::DataType dtype :
       {TensorProto::INT4, TensorProto::UINT4, TensorProto::FLOAT4E2M1}) {
    TensorProto t;
    t.set_name("w");
    t.set_data_type(dtype);
    t.add_dims(10); // ceil(10/8) = 2
    t.add_int32_data(0);
    EXPECT_THROW(VerifyTensor(t), std::invalid_argument);

    TensorProto ok;
    ok.set_name("w");
    ok.set_data_type(dtype);
    ok.add_dims(10);
    ok.add_int32_data(0);
    ok.add_int32_data(0);
    EXPECT_NO_THROW(VerifyTensor(ok));
  }
}

// Packed 2-bit types need ceil(nelem/16) int32 entries.
TEST(onnx_verify, VerifyTensor_Packed2BitInt32DataTooSmall) {
  for (TensorProto::DataType dtype : {TensorProto::INT2, TensorProto::UINT2}) {
    TensorProto t;
    t.set_name("w");
    t.set_data_type(dtype);
    t.add_dims(20); // ceil(20/16) = 2
    t.add_int32_data(0);
    EXPECT_THROW(VerifyTensor(t), std::invalid_argument);

    TensorProto ok;
    ok.set_name("w");
    ok.set_data_type(dtype);
    ok.add_dims(20);
    ok.add_int32_data(0);
    ok.add_int32_data(0);
    EXPECT_NO_THROW(VerifyTensor(ok));
  }
}

TEST(onnx_verify, VerifyAttribute_RefAttrNameOutsideFunction) {
  AttributeProto attr;
  attr.set_name("alpha");
  attr.set_ref_attr_name("alpha");
  EXPECT_THROW(VerifyAttribute(attr, /*in_function_body=*/false, {}), std::invalid_argument);
  EXPECT_NO_THROW(VerifyAttribute(attr, /*in_function_body=*/true, {}));
}

TEST(onnx_verify, VerifyAttribute_TypeMismatch) {
  AttributeProto attr;
  attr.set_name("alpha");
  attr.set_type(AttributeProto::FLOAT);
  // 'f' is not set even though type says FLOAT.
  EXPECT_THROW(VerifyAttribute(attr, false, {}), std::invalid_argument);
}

TEST(onnx_verify, VerifyFunction_Valid) {
  FunctionProto function;
  function.set_name("f");
  function.add_input("a");
  function.add_output("b");
  NodeProto &node = *function.add_node();
  node.set_op_type("Identity");
  node.add_input("a");
  node.add_output("b");
  EXPECT_NO_THROW(VerifyFunction(function));
}

TEST(onnx_verify, VerifyFunction_UnproducedOutput) {
  FunctionProto function;
  function.set_name("f");
  function.add_input("a");
  function.add_output("never_produced");
  EXPECT_THROW(VerifyFunction(function), std::invalid_argument);
}

TEST(onnx_verify, VerifyGraph_SubgraphClosureOverOuterScope) {
  // A control-flow body subgraph may legally reference names already defined
  // in the enclosing graph's scope.
  GraphProto outer;
  outer.set_name("outer");
  std::unordered_set<std::string> outer_scope{"x"};

  GraphProto body;
  body.set_name("body");
  NodeProto *node = body.add_node();
  node->set_op_type("Identity");
  node->add_input("x"); // Defined in outer_scope, not in body.
  node->add_output("y");
  ValueInfoProto *body_output = body.add_output();
  body_output->set_name("y");

  EXPECT_NO_THROW(
      VerifyGraph(body, /*is_main_graph=*/false, /*in_function_body=*/false, &outer_scope));
}
