// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_lib/checker.h"
#include "onnx_op/operator_sets_rt.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

void AddTensorValue(ValueInfoProto &value, const std::string &name) {
  value.set_name(name);
  auto *tensor = value.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape()->add_dim()->set_dim_value(2);
}

FunctionProto MakeIdentityFunction(const std::string &name) {
  FunctionProto function;
  function.set_domain("ai.rt");
  function.set_name(name);
  function.add_input("X");
  function.add_output("Y");
  auto *opset = function.add_opset_import();
  opset->set_domain("test.layer");
  opset->set_version(1);
  NodeProto *identity = function.add_node();
  identity->set_domain("test.layer");
  identity->set_op_type("Pass");
  identity->add_input("X");
  identity->add_output("Y");
  return function;
}

} // namespace

TEST(LayerFunction, MultipleDefinitionsAreStoredAndExecutedAsLocalFunctions) {
  ModelProto model;
  model.set_ir_version(10);
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(18);
  model.add_opset("ai.rt", 1);
  model.add_opset("test.layer", 1);
  *model.add_functions() = MakeIdentityFunction("Layer");
  *model.add_functions() = MakeIdentityFunction("Layer_1");

  GraphProto *graph = model.mutable_graph();
  graph->set_name("layer_functions");
  AddTensorValue(*graph->add_input(), "X");
  AddTensorValue(*graph->add_output(), "Y");
  auto *first = graph->add_node();
  first->set_domain("ai.rt");
  first->set_op_type("Layer");
  first->add_input("X");
  first->add_output("H");
  auto *second = graph->add_node();
  second->set_domain("ai.rt");
  second->set_op_type("Layer_1");
  second->add_input("H");
  second->add_output("Y");

  EXPECT_NO_THROW(checker::check_model(model));
  EXPECT_TRUE(onnx_op::rt::GetAllOnnxOpRtSchemasWithHistory("Layer").empty());

  core::runtime::RuntimeContext context;
  context.RegisterCustomKernel(
      "test.layer", "Pass", [](const NodeProto &node, core::runtime::RuntimeContext &runtime) {
        runtime.Put(node.output(0), runtime.Get(node.input(0)).BorrowView());
      });
  core::runtime::RuntimeSession session(model);
  context.Set("X", core::runtime::Tensor::FromFloat("X", {2}, {1.0f, 2.0f}));
  session.Run(context);
  ASSERT_TRUE(context.Has("Y"));
  EXPECT_EQ(context.Get("Y").AsFloat()[0], 1.0f);
  EXPECT_EQ(context.Get("Y").AsFloat()[1], 2.0f);
}
