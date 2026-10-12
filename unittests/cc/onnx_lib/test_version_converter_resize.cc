// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

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

namespace {

ModelProto MakeResize12Model(int64_t opset, const std::vector<std::string> &inputs,
                             const std::string &coordinate_mode = "") {
  ModelProto model;
  model.set_ir_version(9);
  model.add_opset_import()->set_version(opset);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("resize_conversion");
  for (const std::string &name : {"X", "empty", "sizes", "scales", "roi"}) {
    auto *input = graph->add_input();
    input->set_name(name);
    input->mutable_type()->mutable_tensor_type()->set_elem_type(
        name == "sizes" ? TensorProto::INT64 : TensorProto::FLOAT);
  }
  auto *output = graph->add_output();
  output->set_name("Y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  auto *resize = graph->add_node();
  resize->set_op_type("Resize");
  for (const auto &name : inputs) {
    *resize->add_input() = name;
  }
  *resize->add_output() = "Y";
  if (!coordinate_mode.empty()) {
    auto *attribute = resize->add_attribute();
    attribute->set_name("coordinate_transformation_mode");
    attribute->set_type(AttributeProto::STRING);
    attribute->set_s(coordinate_mode);
  }
  return model;
}

const NodeProto &ConvertedResize(const ModelProto &model) {
  const auto &nodes = model.ref_graph().ref_node();
  return *std::find_if(nodes.begin(), nodes.end(),
                       [](const NodeProto &node) { return node.ref_op_type() == "Resize"; });
}

} // namespace

TEST(onnx_version_converter, Resize12To13RejectsRemovedMode) {
  for (int64_t opset : {11, 12}) {
    for (int64_t target : {13, 18}) {
      EXPECT_THROW(
          version_conversion::ConvertVersion(
              MakeResize12Model(opset, {"X", "roi", "scales"}, "tf_half_pixel_for_nn"), target),
          std::runtime_error);
    }
  }
}

TEST(onnx_version_converter, Resize12To13OmitsScalesWhenSizesProvided) {
  for (int64_t opset : {11, 12}) {
    for (const std::string source : {"input", "initializer", "constant"}) {
      SCOPED_TRACE(source);
      ModelProto model = MakeResize12Model(opset, {"X", "empty", "empty", "sizes"});
      GraphProto *graph = model.mutable_graph();
      if (source == "initializer" || source == "constant") {
        auto &graph_inputs = graph->ref_input();
        graph_inputs.erase(
            std::remove_if(graph_inputs.begin(), graph_inputs.end(),
                           [](const ValueInfoProto &input) { return input.ref_name() == "empty"; }),
            graph_inputs.end());
        TensorProto tensor;
        tensor.set_name("empty");
        tensor.set_data_type(TensorProto::FLOAT);
        *tensor.add_dims() = 0;
        if (source == "initializer") {
          *graph->add_initializer() = tensor;
        } else {
          NodeProto constant;
          constant.set_op_type("Constant");
          *constant.add_output() = "empty";
          auto *attribute = constant.add_attribute();
          attribute->set_name("value");
          attribute->set_type(AttributeProto::TENSOR);
          *attribute->mutable_t() = tensor;
          *graph->add_node() = constant;
          std::iter_swap(graph->ref_node().begin(), graph->ref_node().begin() + 1);
        }
      }
      const ModelProto converted = version_conversion::ConvertVersion(model, 13);
      EXPECT_EQ(converted.ref_opset_import()[0].ref_version(), 13);
      const auto &inputs = ConvertedResize(converted).ref_input();
      ASSERT_EQ(inputs.size(), 4u);
      EXPECT_EQ(inputs[0], "X");
      EXPECT_EQ(inputs[1], "empty");
      EXPECT_EQ(inputs[2], "");
      EXPECT_EQ(inputs[3], "sizes");
    }
  }
}

TEST(onnx_version_converter, Resize12To13PreservesScalesAndSupportedModes) {
  for (int64_t opset : {11, 12}) {
    for (const std::string mode : {"", "half_pixel", "pytorch_half_pixel", "align_corners",
                                   "asymmetric", "tf_crop_and_resize"}) {
      for (const std::vector<std::string> inputs :
           {std::vector<std::string>{"X", "roi", "scales"}, {"X", "roi", "scales", ""}}) {
        SCOPED_TRACE(mode);
        const ModelProto model = MakeResize12Model(opset, inputs, mode);
        const ModelProto converted = version_conversion::ConvertVersion(model, 13);
        const auto &resize = ConvertedResize(converted);
        ASSERT_EQ(resize.ref_input().size(), inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
          EXPECT_EQ(resize.ref_input()[i], inputs[i]);
        }
        EXPECT_EQ(resize.ref_attribute().size(),
                  model.ref_graph().ref_node()[0].ref_attribute().size());
        if (!mode.empty()) {
          EXPECT_EQ(resize.ref_attribute()[0].ref_s(), mode);
        }
      }
    }
  }
}
