// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

TEST(onnx_version_converter, Resize10To11PreservesSemantics) {
  for (const std::string mode : {"", "nearest", "linear"}) {
    SCOPED_TRACE(mode.empty() ? "default mode" : mode);
    ModelProto model;
    model.set_ir_version(9);
    model.add_opset_import()->set_version(10);
    GraphProto *graph = model.mutable_graph();
    graph->set_name("resize_conversion");

    ValueInfoProto *input = graph->add_input();
    input->set_name("X");
    auto *input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(TensorProto::FLOAT);
    for (int64_t dim : {1, 1, 2, 2}) {
      input_type->mutable_shape()->add_dim()->set_dim_value(dim);
    }
    ValueInfoProto *output = graph->add_output();
    output->set_name("Y");
    auto *output_type = output->mutable_type()->mutable_tensor_type();
    output_type->set_elem_type(TensorProto::FLOAT);
    for (int64_t dim : {1, 1, 4, 4}) {
      output_type->mutable_shape()->add_dim()->set_dim_value(dim);
    }

    NodeProto *constant = graph->add_node();
    constant->set_op_type("Constant");
    *constant->add_output() = "scales";
    AttributeProto *value = constant->add_attribute();
    value->set_name("value");
    value->set_type(AttributeProto::TENSOR);
    TensorProto *scales = value->mutable_t();
    scales->set_data_type(TensorProto::FLOAT);
    *scales->add_dims() = 4;
    for (float scale : {1.0f, 1.0f, 2.0f, 2.0f}) {
      *scales->add_float_data() = scale;
    }

    NodeProto *resize = graph->add_node();
    resize->set_op_type("Resize");
    *resize->add_input() = "X";
    *resize->add_input() = "scales";
    *resize->add_output() = "Y";
    if (!mode.empty()) {
      AttributeProto *attribute = resize->add_attribute();
      attribute->set_name("mode");
      attribute->set_type(AttributeProto::STRING);
      attribute->set_s(mode);
    }

    const ModelProto converted = version_conversion::ConvertVersion(model, 11);
    ASSERT_EQ(converted.ref_opset_import().size(), 1u);
    EXPECT_EQ(converted.ref_opset_import()[0].ref_version(), 11);
    const auto &nodes = converted.ref_graph().ref_node();
    const auto node = std::find_if(nodes.begin(), nodes.end(), [](const NodeProto &candidate) {
      return candidate.ref_op_type() == "Resize";
    });
    ASSERT_NE(node, nodes.end());
    ASSERT_EQ(node->ref_input().size(), 3u);
    EXPECT_EQ(node->ref_input()[0], "X");
    EXPECT_EQ(node->ref_input()[2], "scales");

    std::map<std::string, std::string> attributes;
    for (const auto &attribute : node->ref_attribute()) {
      ASSERT_EQ(attribute.ref_type(), AttributeProto::STRING);
      attributes[attribute.ref_name()] = attribute.ref_s();
    }
    EXPECT_EQ(attributes["coordinate_transformation_mode"], "asymmetric");
    if (mode.empty() || mode == "nearest") {
      EXPECT_EQ(attributes["nearest_mode"], "floor");
    } else {
      EXPECT_EQ(attributes.count("nearest_mode"), 0u);
    }
    if (mode.empty()) {
      EXPECT_EQ(attributes.count("mode"), 0u);
    } else {
      EXPECT_EQ(attributes["mode"], mode);
    }
  }
}
