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

} // namespace

TEST(onnx_verify, VerifyModel_Valid) {
  ModelProto model = MakeValidModel();
  EXPECT_NO_THROW(VerifyModel(model));
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
  auto *binding = model.mutable_graph()->add_persistent_bindings();
  binding->set_input_name("state.in");
  binding->set_output_name("state.out");
  EXPECT_NO_THROW(VerifyPersistentBindings(nullptr, model.graph()));
  binding->set_input_name("state.in.cache");
  EXPECT_THROW(VerifyPersistentBindings(nullptr, model.graph()), std::invalid_argument);
  binding->set_input_name("state.in");
  binding->set_output_name("state.out.cache");
  EXPECT_THROW(VerifyPersistentBindings(nullptr, model.graph()), std::invalid_argument);
  binding->set_output_name("state.out");
  auto *input = model.mutable_graph()->add_input();
  input->set_name("other");
  *input->mutable_type() = model.graph().input(0).type();
  auto *second = model.mutable_graph()->add_persistent_bindings();
  second->set_input_name("other");
  second->set_output_name("state.out");
  EXPECT_THROW(VerifyPersistentBindings(nullptr, model.graph()), std::invalid_argument);
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
