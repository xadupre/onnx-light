// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/checker.h"
#include "onnx_lib/common/ir_pb_converter.h"
#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

#include <string>
#include <tuple>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

void AddTensorValue(ValueInfoProto *value, const std::string &name, int32_t elem_type,
                    bool scalar = false) {
  value->set_name(name);
  auto *type = value->mutable_type()->mutable_tensor_type();
  type->set_elem_type(elem_type);
  auto *shape = type->mutable_shape();
  if (!scalar) {
    shape->add_dim()->set_dim_value(1);
  }
}

ModelProto MakeSparseInitializerModel(bool used, bool nested) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(13);
  auto *graph = model.mutable_graph();
  graph->set_name("sparse");
  AddTensorValue(graph->add_input(), "X", TensorProto::FLOAT);
  AddTensorValue(graph->add_output(), nested ? "Z" : "Y", TensorProto::FLOAT);
  if (nested) {
    AddTensorValue(graph->add_input(), "cond", TensorProto::BOOL, true);
    auto *node = graph->add_node();
    node->set_op_type("If");
    node->add_input("cond");
    node->add_output("Z");
    for (const char *name : {"then_branch", "else_branch"}) {
      auto *attr = node->add_attribute();
      attr->set_name(name);
      attr->set_type(AttributeProto::GRAPH);
      auto *branch = attr->mutable_g();
      branch->set_name(name);
      AddTensorValue(branch->add_output(), "Y", TensorProto::FLOAT);
      auto *identity = branch->add_node();
      identity->set_op_type("Identity");
      identity->add_input("X");
      identity->add_output("Y");
    }
    graph = node->mutable_attribute(0)->mutable_g();
  } else {
    auto *identity = graph->add_node();
    identity->set_op_type("Identity");
    identity->add_input("X");
    identity->add_output("Y");
  }
  if (used) {
    *graph->mutable_node(0)->mutable_input(0) = "S";
  }
  auto *sparse = graph->add_sparse_initializer();
  sparse->add_dims(1);
  auto *values = sparse->mutable_values();
  values->set_name("S");
  values->set_data_type(TensorProto::FLOAT);
  values->add_dims(1);
  values->add_float_data(2.0f);
  auto *indices = sparse->mutable_indices();
  indices->set_name("indices");
  indices->set_data_type(TensorProto::INT64);
  indices->add_dims(1);
  indices->add_int64_data(0);
  return model;
}

class VersionConverterSparse : public testing::TestWithParam<std::tuple<bool, bool, int>> {};

TEST_P(VersionConverterSparse, RejectsSparseInitializersWithoutChangingSource) {
  const auto [used, nested, target_version] = GetParam();
  const ModelProto original = MakeSparseInitializerModel(used, nested);
  ASSERT_NO_THROW(checker::check_model(original));
  const auto before = original.SerializeAsString();
  try {
    version_conversion::ConvertVersion(original, target_version);
    FAIL() << "Conversion must reject sparse initializers";
  } catch (const ONNX_LIGHT_NAMESPACE::ConvertError &error) {
    EXPECT_NE(std::string(error.what())
                  .find("Sparse initializers are not supported by the version converter."),
              std::string::npos);
  }
  EXPECT_EQ(original.SerializeAsString(), before);
}

INSTANTIATE_TEST_SUITE_P(onnx_version_converter, VersionConverterSparse,
                         testing::Combine(testing::Bool(), testing::Bool(),
                                          testing::Values(12, 13, 14)));

} // namespace
