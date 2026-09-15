// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto MakeOptionalModel(int64_t opset, TensorProto::DataType data_type) {
  ModelProto model;
  model.set_ir_version(14);
  model.add_opset_import()->set_version(opset);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("optional_conversion");

  ValueInfoProto *input = graph->add_input();
  input->set_name("input");
  input->mutable_type()->mutable_tensor_type()->set_elem_type(data_type);
  ValueInfoProto *output = graph->add_output();
  output->set_name("output");
  output->mutable_type()
      ->mutable_optional_type()
      ->mutable_elem_type()
      ->mutable_tensor_type()
      ->set_elem_type(data_type);

  NodeProto *node = graph->add_node();
  node->set_op_type("Optional");
  *node->add_input() = "input";
  *node->add_output() = "output";
  return model;
}

ModelProto MakeBareOptionalGetElementModel() {
  ModelProto model;
  model.set_ir_version(14);
  model.add_opset_import()->set_version(18);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("optional_get_element_conversion");
  ValueInfoProto *input = graph->add_input();
  input->set_name("input");
  input->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  ValueInfoProto *output = graph->add_output();
  output->set_name("output");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  NodeProto *node = graph->add_node();
  node->set_op_type("OptionalGetElement");
  *node->add_input() = "input";
  *node->add_output() = "output";
  return model;
}

} // namespace

TEST(onnx_version_converter, OptionalOpset28DowngradeAcceptsLegacyType) {
  const ModelProto converted =
      version_conversion::ConvertVersion(MakeOptionalModel(28, TensorProto::FLOAT), 27);
  ASSERT_EQ(converted.ref_opset_import().size(), 1u);
  EXPECT_EQ(converted.ref_opset_import()[0].ref_version(), 27);
}

TEST(onnx_version_converter, OptionalOpset28DowngradeRejectsIr14OnlyType) {
  EXPECT_THROW(
      version_conversion::ConvertVersion(MakeOptionalModel(28, TensorProto::FLOAT6E2M3), 27),
      std::runtime_error);
}

TEST(onnx_version_converter, OptionalGetElementDowngradeRejectsBareInput) {
  EXPECT_THROW(version_conversion::ConvertVersion(MakeBareOptionalGetElementModel(), 17),
               std::runtime_error);
}
