#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/onnx_helper.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace ONNX_LIGHT_NAMESPACE;

TEST(Probe, OptimShapeInferenceWithLocalFunction) {
  ModelProto model;
  model.set_ir_version(8);
  auto *op_ai = model.add_opset_import();
  op_ai->set_domain("");
  op_ai->set_version(14);
  auto *op_local = model.add_opset_import();
  op_local->set_domain("local");
  op_local->set_version(1);

  // Build local function: func_add(a, b) -> c { c = Add(a, b) }
  FunctionProto *f = model.add_functions();
  f->set_name("func_add");
  f->set_domain("local");
  f->add_input("a");
  f->add_input("b");
  f->add_output("c");
  auto *f_opset = f->add_opset_import();
  f_opset->set_domain("");
  f_opset->set_version(14);
  NodeProto *fnode = f->add_node();
  fnode->set_op_type("Add");
  fnode->add_input("a");
  fnode->add_input("b");
  fnode->add_output("c");

  // Build graph: X, Y -> func_add(X, Y) -> Z
  GraphProto *g = model.add_graph();
  g->set_name("g");

  ValueInfoProto *in1 = g->add_input();
  in1->set_name("X");
  TypeProto *t1 = in1->add_type();
  TypeProto::Tensor *tt1 = t1->add_tensor_type();
  tt1->set_elem_type(1); // FLOAT
  TensorShapeProto *s1 = tt1->add_shape();
  s1->add_dim()->set_dim_value(3);

  ValueInfoProto *in2 = g->add_input();
  in2->set_name("Y");
  TypeProto *t2 = in2->add_type();
  TypeProto::Tensor *tt2 = t2->add_tensor_type();
  tt2->set_elem_type(1);
  TensorShapeProto *s2 = tt2->add_shape();
  s2->add_dim()->set_dim_value(3);

  ValueInfoProto *out = g->add_output();
  out->set_name("Z");
  out->add_type();

  NodeProto *node = g->add_node();
  node->set_op_type("func_add");
  node->set_domain("local");
  node->add_input("X");
  node->add_input("Y");
  node->add_output("Z");

  try {
    onnx_optim::shapes::InferShapesModel(model);
    std::cout << "OK output dims: " << out->type().tensor_type().shape().dim_size() << std::endl;
  } catch (const std::exception &e) {
    std::cout << "EXCEPTION: " << e.what() << std::endl;
    FAIL() << e.what();
  }
}
