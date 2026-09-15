// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto MakeReduceLogModel(const std::string &op_type, int64_t opset,
                              TensorProto::DataType data_type) {
  ModelProto model;
  model.set_ir_version(14);
  model.add_opset_import()->set_version(opset);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("reduce_log_conversion");

  for (const auto &[name, type] :
       {std::pair{"data", data_type}, std::pair{"axes", TensorProto::INT64}}) {
    ValueInfoProto *input = graph->add_input();
    input->set_name(name);
    input->mutable_type()->mutable_tensor_type()->set_elem_type(type);
  }
  ValueInfoProto *output = graph->add_output();
  output->set_name("reduced");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(data_type);

  NodeProto *node = graph->add_node();
  node->set_op_type(op_type);
  *node->add_input() = "data";
  *node->add_input() = "axes";
  *node->add_output() = "reduced";
  return model;
}

} // namespace

TEST(onnx_version_converter, ReduceLogOpset28UpgradeAcceptsFloatDataAndInt64Axes) {
  for (const std::string &op : {std::string("ReduceLogSum"), std::string("ReduceLogSumExp")}) {
    const ModelProto converted =
        version_conversion::ConvertVersion(MakeReduceLogModel(op, 27, TensorProto::FLOAT), 28);
    ASSERT_EQ(converted.ref_opset_import().size(), 1u);
    EXPECT_EQ(converted.ref_opset_import()[0].ref_version(), 28);
  }
}

TEST(onnx_version_converter, ReduceLogOpset28UpgradeRejectsIntegerData) {
  for (const std::string &op : {std::string("ReduceLogSum"), std::string("ReduceLogSumExp")}) {
    EXPECT_THROW(
        version_conversion::ConvertVersion(MakeReduceLogModel(op, 27, TensorProto::INT64), 28),
        std::runtime_error)
        << op;
  }
}
