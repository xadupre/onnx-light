// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Verifies that doc_string on graph inputs, outputs and value_info survives a
// round-trip through the internal IR (ImportModelProto followed by
// ExportModelProto). This mirrors the behavior added upstream in
// https://github.com/onnx/onnx/pull/8205.

#include "../common/ir.h"
#include "../common/ir_pb_converter.h"
#include "onnx.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto BuildModelWithDocStrings() {
  ModelProto model;
  model.set_ir_version(7);
  OperatorSetIdProto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  GraphProto *graph = model.add_graph();
  graph->set_name("test");

  // Graph input: x with a doc_string.
  ValueInfoProto *input = graph->add_input();
  input->set_name("x");
  input->set_doc_string("input doc");
  TypeProto *input_type = input->add_type();
  TypeProto::Tensor *input_tensor = input_type->add_tensor_type();
  input_tensor->set_elem_type(static_cast<int32_t>(TensorProto::DataType::FLOAT));

  // Node producing an intermediate value: temp = Neg(x).
  NodeProto *node1 = graph->add_node();
  node1->set_op_type("Neg");
  *node1->add_input() = "x";
  *node1->add_output() = "temp";

  // Node producing the output: y = Neg(temp).
  NodeProto *node2 = graph->add_node();
  node2->set_op_type("Neg");
  *node2->add_input() = "temp";
  *node2->add_output() = "y";

  // value_info for the intermediate value, with a doc_string.
  ValueInfoProto *vi = graph->add_value_info();
  vi->set_name("temp");
  vi->set_doc_string("value_info doc");
  TypeProto *vi_type = vi->add_type();
  TypeProto::Tensor *vi_tensor = vi_type->add_tensor_type();
  vi_tensor->set_elem_type(static_cast<int32_t>(TensorProto::DataType::FLOAT));

  // Graph output: y with a doc_string.
  ValueInfoProto *output = graph->add_output();
  output->set_name("y");
  output->set_doc_string("output doc");
  TypeProto *output_type = output->add_type();
  TypeProto::Tensor *output_tensor = output_type->add_tensor_type();
  output_tensor->set_elem_type(static_cast<int32_t>(TensorProto::DataType::FLOAT));

  return model;
}

} // namespace

TEST(onnx_ir_pb_converter, PreserveValueDocString) {
  ModelProto model = BuildModelWithDocStrings();

  std::shared_ptr<Graph> g = ImportModelProto(model);
  ASSERT_TRUE(g != nullptr);

  ModelProto exported;
  exported.set_ir_version(model.ir_version());
  ExportModelProto(&exported, g);

  const GraphProto &gp = exported.ref_graph();

  bool checked_input = false;
  for (const auto &input : gp.ref_input()) {
    if (input.name() == "x") {
      EXPECT_TRUE(input.has_doc_string());
      EXPECT_EQ(input.doc_string(), std::string("input doc"));
      checked_input = true;
    }
  }
  EXPECT_TRUE(checked_input);

  bool checked_output = false;
  for (const auto &output : gp.ref_output()) {
    if (output.name() == "y") {
      EXPECT_TRUE(output.has_doc_string());
      EXPECT_EQ(output.doc_string(), std::string("output doc"));
      checked_output = true;
    }
  }
  EXPECT_TRUE(checked_output);

  bool checked_value_info = false;
  for (const auto &vi : gp.ref_value_info()) {
    if (vi.name() == "temp") {
      EXPECT_TRUE(vi.has_doc_string());
      EXPECT_EQ(vi.doc_string(), std::string("value_info doc"));
      checked_value_info = true;
    }
  }
  EXPECT_TRUE(checked_value_info);
}

TEST(onnx_ir_pb_converter, PreserveFloat8E8M0InitializerData) {
  ModelProto model;
  model.set_ir_version(13);
  GraphProto *graph = model.add_graph();
  graph->set_name("float8e8m0");
  TensorProto *initializer = graph->add_initializer();
  initializer->set_name("scale");
  initializer->set_data_type(TensorProto::FLOAT8E8M0);
  initializer->add_dims(2);
  initializer->add_int32_data(127);
  initializer->add_int32_data(128);

  std::shared_ptr<Graph> imported = ImportModelProto(model);
  ASSERT_TRUE(imported != nullptr);

  ModelProto exported;
  exported.set_ir_version(model.ir_version());
  ExportModelProto(&exported, imported);

  ASSERT_EQ(exported.ref_graph().ref_initializer().size(), 1u);
  const TensorProto &exported_initializer = exported.ref_graph().ref_initializer()[0];
  EXPECT_EQ(exported_initializer.data_type(), TensorProto::FLOAT8E8M0);
  ASSERT_EQ(exported_initializer.ref_int32_data().size(), 2u);
  EXPECT_EQ(exported_initializer.ref_int32_data()[0], 127);
  EXPECT_EQ(exported_initializer.ref_int32_data()[1], 128);
}
