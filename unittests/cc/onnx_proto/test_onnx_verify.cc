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
