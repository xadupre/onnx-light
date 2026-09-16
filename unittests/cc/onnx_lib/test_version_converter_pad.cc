// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto MakePadModel(int64_t opset, TensorProto::DataType data_type = TensorProto::FLOAT,
                        const std::vector<std::string> &inputs = {"data", "pads"},
                        const std::string &mode = "") {
  ModelProto model;
  model.set_ir_version(9);
  model.add_opset_import()->set_version(opset);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("pad_conversion");

  for (const auto &[name, type] :
       {std::pair{"data", data_type}, std::pair{"pads", TensorProto::INT64},
        std::pair{"constant_value", data_type}, std::pair{"axes", TensorProto::INT64}}) {
    if (std::find(inputs.begin(), inputs.end(), name) != inputs.end()) {
      ValueInfoProto *input = graph->add_input();
      input->set_name(name);
      input->mutable_type()->mutable_tensor_type()->set_elem_type(type);
    }
  }
  ValueInfoProto *output = graph->add_output();
  output->set_name("output");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(data_type);

  NodeProto *node = graph->add_node();
  node->set_op_type("Pad");
  for (const auto &input : inputs) {
    *node->add_input() = input;
  }
  *node->add_output() = "output";
  if (!mode.empty()) {
    AttributeProto *attribute = node->add_attribute();
    attribute->set_name("mode");
    attribute->set_type(AttributeProto::STRING);
    attribute->set_s(mode);
  }
  return model;
}

} // namespace

TEST(onnx_version_converter, Pad18To17WithoutAxes) {
  for (const std::vector<std::string> &inputs : {std::vector<std::string>{"data", "pads"},
                                                 {"data", "pads", "constant_value"},
                                                 {"data", "pads", "", ""},
                                                 {"data", "pads", "constant_value", ""}}) {
    const ModelProto converted =
        version_conversion::ConvertVersion(MakePadModel(18, TensorProto::FLOAT, inputs), 17);
    EXPECT_EQ(converted.ref_opset_import()[0].ref_version(), 17);
    ASSERT_EQ(converted.ref_graph().ref_node().size(), 1u);
    const auto &node = converted.ref_graph().ref_node()[0];
    const size_t expected_size = inputs.size() == 4 ? 3 : inputs.size();
    ASSERT_EQ(node.ref_input().size(), expected_size);
    for (size_t i = 0; i < expected_size; ++i) {
      EXPECT_EQ(node.ref_input()[i], inputs[i]);
    }
  }
}

TEST(onnx_version_converter, Pad18To17RejectsAxes) {
  EXPECT_THROW(version_conversion::ConvertVersion(
                   MakePadModel(18, TensorProto::FLOAT, {"data", "pads", "", "axes"}), 17),
               std::runtime_error);
}

TEST(onnx_version_converter, Pad19To18PreservesSupportedModes) {
  for (const std::string mode : {"", "constant", "reflect", "edge"}) {
    const ModelProto converted = version_conversion::ConvertVersion(
        MakePadModel(19, TensorProto::FLOAT, {"data", "pads"}, mode), 18);
    EXPECT_EQ(converted.ref_opset_import()[0].ref_version(), 18);
    ASSERT_EQ(converted.ref_graph().ref_node().size(), 1u);
    const auto &node = converted.ref_graph().ref_node()[0];
    if (mode.empty()) {
      EXPECT_TRUE(node.ref_attribute().empty());
    } else {
      ASSERT_EQ(node.ref_attribute().size(), 1u);
      EXPECT_EQ(node.ref_attribute()[0].ref_name(), "mode");
      EXPECT_EQ(node.ref_attribute()[0].ref_s(), mode);
    }
  }
}

TEST(onnx_version_converter, Pad19To18RejectsWrap) {
  EXPECT_THROW(version_conversion::ConvertVersion(
                   MakePadModel(19, TensorProto::FLOAT, {"data", "pads"}, "wrap"), 18),
               std::runtime_error);
}

TEST(onnx_version_converter, Pad13To12AcceptsSupportedTypes) {
  for (const auto type :
       {TensorProto::FLOAT, TensorProto::FLOAT16, TensorProto::DOUBLE, TensorProto::INT8,
        TensorProto::INT16, TensorProto::INT32, TensorProto::INT64, TensorProto::UINT8,
        TensorProto::UINT16, TensorProto::UINT32, TensorProto::UINT64}) {
    const ModelProto converted = version_conversion::ConvertVersion(MakePadModel(13, type), 12);
    EXPECT_EQ(converted.ref_opset_import()[0].ref_version(), 12);
    EXPECT_EQ(converted.ref_graph().ref_input()[0].ref_type().ref_tensor_type().ref_elem_type(),
              type);
  }
}

TEST(onnx_version_converter, Pad13To12RejectsUnsupportedTypes) {
  for (const auto type : {TensorProto::BFLOAT16, TensorProto::BOOL, TensorProto::COMPLEX64,
                          TensorProto::COMPLEX128, TensorProto::STRING}) {
    EXPECT_THROW(version_conversion::ConvertVersion(MakePadModel(13, type), 12), std::runtime_error)
        << type;
  }
}
