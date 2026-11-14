// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

void SetCaptureTensor(ValueInfoProto *value, const std::string &name, TensorProto::DataType type,
                      int64_t size) {
  value->set_name(name);
  auto *tensor = value->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(type);
  tensor->mutable_shape()->add_dim()->set_dim_value(size);
}

void AddCaptureInitializer(GraphProto *graph, TensorProto::DataType type, int64_t size) {
  auto *initializer = graph->add_initializer();
  initializer->set_name("X");
  initializer->set_data_type(type);
  initializer->add_dims(size);
  for (int64_t i = 0; i < size; ++i) {
    if (type == TensorProto::DOUBLE)
      initializer->add_double_data(1.0);
    else
      initializer->add_float_data(1.0f);
  }
}

GraphProto *AddCaptureCarrier(GraphProto *graph, bool graph_list = false) {
  auto *node = graph->add_node();
  node->set_domain("custom");
  node->set_op_type("Carrier");
  node->add_output("Y");
  auto *attribute = node->add_attribute();
  attribute->set_name("body");
  attribute->set_type(graph_list ? AttributeProto::GRAPHS : AttributeProto::GRAPH);
  auto *body = graph_list ? attribute->add_graphs() : attribute->mutable_g();
  body->set_name("capturing_body");
  body->add_output()->set_name("B");
  return body;
}

void AddCapturedIdentity(GraphProto *graph, const std::string &output = "B") {
  auto *identity = graph->add_node();
  identity->set_op_type("Identity");
  identity->add_input("X");
  identity->add_output(output);
}

ModelProto MakeCaptureModel() {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(13);
  auto *custom = model.add_opset_import();
  custom->set_domain("custom");
  custom->set_version(1);
  auto *graph = model.mutable_graph();
  graph->set_name("capture_metadata");
  SetCaptureTensor(graph->add_input(), "X", TensorProto::DOUBLE, 3);
  graph->add_output()->set_name("Y");
  return model;
}

void ExpectCaptureOutput(const GraphProto &graph, TensorProto::DataType type, int64_t size) {
  const auto &tensor = graph.output(0).type().tensor_type();
  EXPECT_EQ(tensor.elem_type(), type);
  ASSERT_EQ(tensor.shape().dim_size(), 1);
  EXPECT_EQ(tensor.shape().dim(0).dim_value(), size);
}

TEST(onnx_version_converter, CapturesPreserveExistingTypeAndShapeAndFillOnlyMissingFields) {
  for (const std::string annotation : {"complete", "type_only", "shape_only", "symbolic"}) {
    SCOPED_TRACE(annotation);
    auto model = MakeCaptureModel();
    auto *body = AddCaptureCarrier(model.mutable_graph());
    auto *capture = body->add_value_info();
    SetCaptureTensor(capture, "X", TensorProto::FLOAT, 7);
    if (annotation == "type_only")
      capture->mutable_type()->mutable_tensor_type()->clear_shape();
    if (annotation == "shape_only")
      capture->mutable_type()->mutable_tensor_type()->clear_elem_type();
    if (annotation == "symbolic") {
      auto *dimension =
          capture->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0);
      dimension->clear_dim_value();
      dimension->set_dim_param("captured_dimension");
    }
    AddCapturedIdentity(body);
    const auto original = model.SerializeAsString();
    const auto converted = version_conversion::ConvertVersion(model, 14);
    const auto &converted_body = converted.graph().node(0).attribute(0).g();
    EXPECT_EQ(model.SerializeAsString(), original);
    if (annotation == "symbolic") {
      const auto &tensor = converted_body.output(0).type().tensor_type();
      EXPECT_EQ(tensor.elem_type(), TensorProto::FLOAT);
      ASSERT_EQ(tensor.shape().dim_size(), 1);
      EXPECT_EQ(tensor.shape().dim(0).dim_param(), "captured_dimension");
    } else {
      ExpectCaptureOutput(converted_body,
                          annotation == "shape_only" ? TensorProto::DOUBLE : TensorProto::FLOAT,
                          annotation == "type_only" ? 3 : 7);
    }
  }
}

TEST(onnx_version_converter, CapturesInheritInitializerOnlyTensorMetadataForGenericOperators) {
  auto model = MakeCaptureModel();
  model.mutable_graph()->clear_input();
  AddCaptureInitializer(model.mutable_graph(), TensorProto::DOUBLE, 3);
  auto *body = AddCaptureCarrier(model.mutable_graph(), true);
  AddCapturedIdentity(body);
  const auto converted = version_conversion::ConvertVersion(model, 14);
  const auto &converted_body = converted.graph().node(0).attribute(0).graphs(0);
  ExpectCaptureOutput(converted_body, TensorProto::DOUBLE, 3);
}

TEST(onnx_version_converter, NestedCapturesUseNearestAnnotationOrLocalInitializer) {
  for (const bool initializer : {false, true}) {
    auto model = MakeCaptureModel();
    auto *body = AddCaptureCarrier(model.mutable_graph());
    if (initializer)
      AddCaptureInitializer(body, TensorProto::FLOAT, 7);
    else
      SetCaptureTensor(body->add_value_info(), "X", TensorProto::FLOAT, 7);
    AddCapturedIdentity(body, "local_use");
    auto *nested = AddCaptureCarrier(body, true);
    body->mutable_output(0)->set_name("Y");
    AddCapturedIdentity(nested);
    const auto converted = version_conversion::ConvertVersion(model, 14);
    const auto &converted_body = converted.graph().node(0).attribute(0).g();
    const auto &converted_nested = converted_body.node(1).attribute(0).graphs(0);
    ExpectCaptureOutput(converted_nested, TensorProto::FLOAT, 7);
  }
}

TEST(onnx_version_converter, CapturedGraphOutputsPreserveTheirOwnAnnotations) {
  auto model = MakeCaptureModel();
  auto *body = AddCaptureCarrier(model.mutable_graph());
  body->clear_output();
  auto *output = body->add_output();
  SetCaptureTensor(output, "X", TensorProto::FLOAT, 7);
  output->set_doc_string("explicit capture output");
  const auto converted = version_conversion::ConvertVersion(model, 14);
  const auto &converted_body = converted.graph().node(0).attribute(0).g();
  ExpectCaptureOutput(converted_body, TensorProto::FLOAT, 7);
  EXPECT_EQ(converted_body.output(0).doc_string(), "explicit capture output");
}

} // namespace
