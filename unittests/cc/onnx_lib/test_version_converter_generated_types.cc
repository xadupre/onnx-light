// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

void AddTensorValue(ValueInfoProto *value, const std::string &name, TensorProto::DataType type) {
  value->set_name(name);
  auto *tensor_type = value->mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(type);
  tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
}

ModelProto MakeConstantConsumerModel(int64_t opset, const std::string &consumer) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(opset);
  auto *graph = model.mutable_graph();
  graph->set_name("constant_consumer");

  if (consumer == "Add")
    AddTensorValue(graph->add_input(), "X", TensorProto::FLOAT);
  AddTensorValue(graph->add_output(), "Y", TensorProto::FLOAT);

  auto *constant = graph->add_node();
  constant->set_op_type("Constant");
  *constant->add_output() = "C";
  auto *value = constant->add_attribute();
  value->set_name("value");
  value->set_type(AttributeProto::TENSOR);
  value->mutable_t()->set_data_type(TensorProto::FLOAT);
  value->mutable_t()->add_float_data(1.0f);

  auto *node = graph->add_node();
  node->set_op_type(consumer);
  *node->add_input() = consumer == "Add" ? "X" : "C";
  if (consumer == "Add")
    *node->add_input() = "C";
  *node->add_output() = "Y";
  return model;
}

ModelProto MakeCustomDomainConstantModel() {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(10);
  model.add_opset_import()->set_domain("custom");
  model.mutable_opset_import(1)->set_version(1);

  auto *graph = model.mutable_graph();
  graph->set_name("custom_domain_constant");
  graph->add_output()->set_name("Y");
  auto *constant = graph->add_node();
  constant->set_domain("custom");
  constant->set_op_type("Constant");
  constant->add_output("Y");
  auto *value = constant->add_attribute();
  value->set_name("value");
  value->set_type(AttributeProto::TENSOR);
  value->mutable_t()->set_data_type(TensorProto::FLOAT);
  value->mutable_t()->add_float_data(1.0f);
  return model;
}

ModelProto MakeFunctionAdapterModel(const std::string &op, TensorProto::DataType type,
                                    bool annotate_input = true) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(10);
  model.add_opset_import()->set_domain("local");
  model.mutable_opset_import(1)->set_version(1);

  auto *graph = model.mutable_graph();
  graph->set_name("function_adapter");
  AddTensorValue(graph->add_input(), "X", type);
  AddTensorValue(graph->add_output(), "Y", type);
  auto *call = graph->add_node();
  call->set_domain("local");
  call->set_op_type(op);
  *call->add_input() = "X";
  *call->add_output() = "Y";

  auto *function = model.add_functions();
  function->set_domain("local");
  function->set_name(op);
  function->add_input("X");
  function->add_output("Y");
  function->add_opset_import()->set_version(10);
  if (annotate_input)
    AddTensorValue(function->add_value_info(), "X", type);
  AddTensorValue(function->add_value_info(), "Y", type);

  auto *node = function->add_node();
  node->set_op_type(op == "LocalClip" ? "Clip" : "Pad");
  *node->add_input() = "X";
  *node->add_output() = "Y";
  if (op == "LocalClip") {
    auto *minimum = node->add_attribute();
    minimum->set_name("min");
    minimum->set_type(AttributeProto::FLOAT);
    minimum->set_f(0.5f);
  } else {
    auto *pads = node->add_attribute();
    pads->set_name("pads");
    pads->set_type(AttributeProto::INTS);
    pads->add_ints(0);
    pads->add_ints(0);
    auto *value = node->add_attribute();
    value->set_name("value");
    value->set_type(AttributeProto::FLOAT);
    value->set_f(0.5f);
  }
  return model;
}

void ExpectFunctionAdapterConstantType(const std::string &op, TensorProto::DataType type) {
  const ModelProto converted =
      version_conversion::ConvertVersion(MakeFunctionAdapterModel(op, type), 11);
  const auto &function = converted.functions(0);
  const NodeProto *adapted = nullptr;
  const NodeProto *cast = nullptr;
  for (const auto &node : function.node()) {
    if (node.op_type() == (op == "LocalClip" ? "Clip" : "Pad"))
      adapted = &node;
    if (node.op_type() == "Cast")
      cast = &node;
  }
  ASSERT_NE(adapted, nullptr);
  ASSERT_NE(cast, nullptr);
  const int value_input = op == "LocalClip" ? 1 : 2;
  ASSERT_GT(adapted->input_size(), value_input);
  ASSERT_EQ(cast->attribute_size(), 1);
  EXPECT_EQ(cast->attribute(0).name(), "to");
  EXPECT_EQ(cast->attribute(0).i(), type);
  EXPECT_EQ(adapted->input(value_input), cast->output(0));
}

ModelProto MakeGroupNormalizationModel(bool annotate_inputs) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(21);
  auto *graph = model.mutable_graph();
  graph->set_name("group_normalization_conversion");
  for (const char *name : {"X", "scale", "bias"}) {
    if (annotate_inputs) {
      AddTensorValue(graph->add_input(), name, TensorProto::FLOAT);
    } else {
      graph->add_input()->set_name(name);
    }
  }
  AddTensorValue(graph->add_output(), "Y", TensorProto::FLOAT);
  auto *node = graph->add_node();
  node->set_op_type("GroupNormalization");
  node->add_input("X");
  node->add_input("scale");
  node->add_input("bias");
  node->add_output("Y");
  auto *groups = node->add_attribute();
  groups->set_name("num_groups");
  groups->set_type(AttributeProto::INT);
  groups->set_i(1);
  return model;
}

TEST(onnx_version_converter, InfersConstantOutputTypeForAdaptersAndConsumers) {
  EXPECT_NO_THROW(version_conversion::ConvertVersion(MakeConstantConsumerModel(9, "Add"), 8));
  EXPECT_NO_THROW(
      version_conversion::ConvertVersion(MakeConstantConsumerModel(19, "Identity"), 18));
}

TEST(onnx_version_converter, DoesNotInferCustomDomainConstantOutputType) {
  const ModelProto converted =
      version_conversion::ConvertVersion(MakeCustomDomainConstantModel(), 9);
  EXPECT_EQ(converted.graph().output(0).type().tensor_type().elem_type(), TensorProto::UNDEFINED);
}

TEST(onnx_version_converter, RejectsFunctionConversionWithoutRequiredConstantType) {
  for (const char *op : {"LocalClip", "LocalPad"}) {
    EXPECT_THROW(version_conversion::ConvertVersion(
                     MakeFunctionAdapterModel(op, TensorProto::DOUBLE, false), 11),
                 std::runtime_error)
        << op;
  }
}

TEST(onnx_version_converter, ConvertsClipFunctionBoundsToInputType) {
  for (const auto type : {TensorProto::FLOAT16, TensorProto::DOUBLE})
    ExpectFunctionAdapterConstantType("LocalClip", type);
}

TEST(onnx_version_converter, ConvertsPadFunctionValueToInputType) {
  for (const auto type : {TensorProto::FLOAT16, TensorProto::DOUBLE})
    ExpectFunctionAdapterConstantType("LocalPad", type);
}

TEST(onnx_version_converter, ConvertsGroupNormalizationAcrossMultipleDowngrades) {
  EXPECT_NO_THROW(version_conversion::ConvertVersion(MakeGroupNormalizationModel(true), 19));
}

TEST(onnx_version_converter, RejectsGroupNormalizationWithUnknownInputType) {
  EXPECT_THROW(version_conversion::ConvertVersion(MakeGroupNormalizationModel(false), 19),
               std::runtime_error);
}

} // namespace
