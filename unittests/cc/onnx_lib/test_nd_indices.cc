// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/checker.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_lib/version_converter/convert.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

void AddNDInput(GraphProto &graph, const std::string &name, int dtype,
                const std::vector<int64_t> &dims) {
  auto *input = graph.add_input();
  input->set_name(name);
  auto *type = input->mutable_type()->mutable_tensor_type();
  type->set_elem_type(dtype);
  auto *shape = type->mutable_shape();
  for (int64_t size : dims) {
    auto *dim = shape->add_dim();
    if (size == -1) {
      dim->set_dim_param("B");
    } else {
      dim->set_dim_value(size);
    }
  }
}

ModelProto MakeNDModel(const std::string &op, int index_type, int version,
                       const std::vector<int64_t> &data_shape = {2, 3},
                       const std::vector<int64_t> &indices_shape = {1, 1}, int64_t batch_dims = 0) {
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  model.add_opset_import()->set_version(version);
  auto *graph = model.mutable_graph();
  graph->set_name("nd_indices");
  AddNDInput(*graph, "data", TensorProto::FLOAT, data_shape);
  AddNDInput(*graph, "indices", index_type, indices_shape);
  auto *node = graph->add_node();
  node->set_op_type(op);
  node->add_input("data");
  node->add_input("indices");
  node->add_output("output");
  if (op == "GatherND") {
    if (version >= 12) {
      AddAttribute<int64_t>(*node, "batch_dims", batch_dims);
    }
  } else {
    std::vector<int64_t> updates_shape(indices_shape.begin(), indices_shape.end() - 1);
    updates_shape.insert(updates_shape.end(), data_shape.begin() + indices_shape.back(),
                         data_shape.end());
    AddNDInput(*graph, "updates", TensorProto::FLOAT, updates_shape);
    node->add_input("updates");
    if (version >= 16) {
      AddAttribute<std::string>(*node, "reduction", "add");
    }
  }
  graph->add_output()->set_name("output");
  return model;
}

void InferNDModel(ModelProto &model) {
  shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                               ShapeInferenceOptions(true, 1, false));
}

} // namespace

TEST(NDIndices, TypeConstraintsPreserveOlderOpsets) {
  RegisterAllOnnxOperatorSchemas();
  for (const std::string op : {"GatherND", "ScatterND"}) {
    for (int version : {11, 12, 13, 16, 18, 28, 29}) {
      for (int dtype : {TensorProto::INT32, TensorProto::INT64, TensorProto::FLOAT}) {
        SCOPED_TRACE(op + ":" + std::to_string(version) + ":" + std::to_string(dtype));
        auto model = MakeNDModel(op, dtype, version);
        if (dtype == TensorProto::FLOAT || (version < 29 && dtype == TensorProto::INT32)) {
          EXPECT_THROW(InferNDModel(model), InferenceError);
        } else {
          ASSERT_NO_THROW(InferNDModel(model));
          EXPECT_NO_THROW(checker::check_model(model, true));
        }
      }
    }
  }
}

TEST(NDIndices, GatherShapeInference) {
  RegisterAllOnnxOperatorSchemas();
  struct ShapeCase {
    std::vector<int64_t> data, indices, output;
    int64_t batch_dims;
  };
  const ShapeCase cases[] = {
      {{-1, 4, 3}, {-1, 1, 2, 1}, {-1, 1, 2, 3}, 1},
      {{0, 4, 3}, {0, 2, 1}, {0, 2, 3}, 1},
      {{2, 4, 3}, {2, 0, 1}, {2, 0, 3}, 1},
      {{2, 3, 4, 5}, {2, 3, 2, 1}, {2, 3, 2, 5}, 2},
      {{2, 3}, {2}, {}, 0},
  };
  for (int version : {13, 29}) {
    for (int dtype : {TensorProto::INT32, TensorProto::INT64}) {
      if (version < 29 && dtype == TensorProto::INT32) {
        continue;
      }
      for (const auto &test : cases) {
        auto model =
            MakeNDModel("GatherND", dtype, version, test.data, test.indices, test.batch_dims);
        ASSERT_NO_THROW(InferNDModel(model));
        const auto &output = model.graph().output(0).type().tensor_type();
        ASSERT_TRUE(output.has_shape());
        ASSERT_EQ(output.shape().dim_size(), test.output.size());
        for (size_t i = 0; i < test.output.size(); ++i) {
          if (test.output[i] == -1) {
            EXPECT_EQ(output.shape().dim(i).dim_param(), "B");
          } else {
            EXPECT_EQ(output.shape().dim(i).dim_value(), test.output[i]);
          }
        }
        EXPECT_NO_THROW(checker::check_model(model, true));
      }
    }
  }
}

TEST(NDIndices, ScatterSymbolicShapeInference) {
  RegisterAllOnnxOperatorSchemas();
  for (int dtype : {TensorProto::INT32, TensorProto::INT64}) {
    auto model = MakeNDModel("ScatterND", dtype, 29, {-1, 3, 2}, {4, 2});
    ASSERT_NO_THROW(InferNDModel(model));
    const auto &shape = model.graph().output(0).type().tensor_type().shape();
    ASSERT_EQ(shape.dim_size(), 3u);
    EXPECT_EQ(shape.dim(0).dim_param(), "B");
    EXPECT_EQ(shape.dim(1).dim_value(), 3);
    EXPECT_EQ(shape.dim(2).dim_value(), 2);
  }
}

TEST(NDIndices, GatherRejectsInvalidShapes) {
  RegisterAllOnnxOperatorSchemas();
  for (int dtype : {TensorProto::INT32, TensorProto::INT64}) {
    for (int64_t batch_dims : std::vector<int64_t>{-1, 3, std::numeric_limits<int64_t>::max()}) {
      auto model = MakeNDModel("GatherND", dtype, 29, {2, 3}, {1, 1}, batch_dims);
      EXPECT_THROW(InferNDModel(model), std::runtime_error);
    }
    auto scalar_data = MakeNDModel("GatherND", dtype, 29, {}, {1, 1});
    EXPECT_THROW(InferNDModel(scalar_data), InferenceError);
    auto scalar_indices = MakeNDModel("GatherND", dtype, 29, {2, 3}, {});
    EXPECT_THROW(InferNDModel(scalar_indices), InferenceError);
  }
}

TEST(NDIndices, DowngradeWidensIndicesWithoutChangingInputsOrAttributes) {
  RegisterAllOnnxOperatorSchemas();
  for (const std::string op : {"GatherND", "ScatterND"}) {
    for (int dtype : {TensorProto::INT32, TensorProto::INT64}) {
      for (const std::string provider : {"input", "initializer", "constant", "unknown"}) {
        SCOPED_TRACE(op + ":" + provider + ":" + std::to_string(dtype));
        auto model = MakeNDModel(op, dtype, 29, {2, 3, 2}, {2, 1}, 1);
        ASSERT_NO_THROW(InferNDModel(model));
        auto *graph = model.mutable_graph();
        const auto attribute = graph->node(0).attribute(0).SerializeAsString();
        if (provider == "initializer" || provider == "constant") {
          TensorProto indices;
          indices.set_name("indices");
          indices.set_data_type(dtype);
          indices.add_dims(2);
          indices.add_dims(1);
          if (dtype == TensorProto::INT32) {
            indices.add_int32_data(0);
            indices.add_int32_data(-1);
          } else {
            indices.add_int64_data(0);
            indices.add_int64_data(-1);
          }
          if (provider == "constant") {
            *graph->add_value_info() = graph->input(1);
          }
          graph->mutable_input()->erase(graph->mutable_input()->begin() + 1);
          if (provider == "initializer") {
            *graph->add_initializer() = indices;
          } else {
            NodeProto constant;
            constant.set_op_type("Constant");
            constant.add_output("indices");
            auto *value = constant.add_attribute();
            value->set_name("value");
            value->set_type(AttributeProto::TENSOR);
            *value->mutable_t() = indices;
            graph->add_node(std::move(constant));
            graph->mutable_node()->SwapElements(0, 1);
          }
        } else if (provider == "unknown") {
          NodeProto opaque;
          opaque.set_op_type("OpaqueIndices");
          opaque.set_domain("test.nd_indices");
          opaque.add_input("indices");
          opaque.add_output("untyped_indices");
          *graph->mutable_node(0)->mutable_input(1) = "untyped_indices";
          graph->add_node(std::move(opaque));
          graph->mutable_node()->SwapElements(0, 1);
          auto *opset = model.add_opset_import();
          opset->set_domain("test.nd_indices");
          opset->set_version(1);
        }
        const auto before = model.SerializeAsString();
        const int target = op == "GatherND" ? 13 : 18;
        auto converted = version_conversion::ConvertVersion(model, target);
        EXPECT_EQ(model.SerializeAsString(), before);
        EXPECT_EQ(converted.opset_import(0).version(), target);
        ASSERT_EQ(converted.graph().input_size(), model.graph().input_size());
        for (int i = 0; i < model.graph().input_size(); ++i) {
          EXPECT_EQ(converted.graph().input(i).SerializeAsString(),
                    model.graph().input(i).SerializeAsString());
        }
        size_t casts = 0;
        std::string cast_output;
        for (const auto &node : converted.graph().node()) {
          if (node.op_type() == "Cast") {
            ++casts;
            EXPECT_EQ(node.input(0), provider == "unknown" ? "untyped_indices" : "indices");
            EXPECT_EQ(node.attribute(0).i(), TensorProto::INT64);
            cast_output = node.output(0);
          } else if (node.op_type() == op) {
            EXPECT_EQ(node.attribute(0).SerializeAsString(), attribute);
            EXPECT_EQ(node.input(1), cast_output.empty() ? "indices" : cast_output);
          }
        }
        EXPECT_EQ(casts, dtype == TensorProto::INT32 || provider == "unknown" ? 1u : 0u);
        EXPECT_NO_THROW(checker::check_model(converted, true));
      }
    }
  }
}

TEST(NDIndices, UpgradePreservesInt64Indices) {
  RegisterAllOnnxOperatorSchemas();
  for (const std::string op : {"GatherND", "ScatterND"}) {
    auto model = MakeNDModel(op, TensorProto::INT64, 28);
    ASSERT_NO_THROW(InferNDModel(model));
    auto converted = version_conversion::ConvertVersion(model, 29);
    EXPECT_EQ(converted.opset_import(0).version(), 29);
    ASSERT_EQ(converted.graph().node_size(), 1u);
    EXPECT_EQ(converted.graph().node(0).SerializeAsString(),
              model.graph().node(0).SerializeAsString());
    EXPECT_NO_THROW(checker::check_model(converted, true));
  }
}

TEST(NDIndices, DowngradeRejectsMissingIndices) {
  RegisterAllOnnxOperatorSchemas();
  for (const std::string op : {"GatherND", "ScatterND"}) {
    auto model = MakeNDModel(op, TensorProto::INT32, 29);
    auto *node = model.mutable_graph()->mutable_node(0);
    node->clear_input();
    node->add_input("data");
    EXPECT_THROW(version_conversion::ConvertVersion(model, 28), std::runtime_error);
  }
}

TEST(NDIndices, CastDowngradeRejectsFloat6Types) {
  RegisterAllOnnxOperatorSchemas();
  for (int dtype : {TensorProto::FLOAT6E2M3, TensorProto::FLOAT6E3M2}) {
    ModelProto model;
    model.set_ir_version(IR_VERSION);
    model.add_opset_import()->set_version(28);
    auto *graph = model.mutable_graph();
    graph->set_name("cast_float6");
    AddNDInput(*graph, "data", dtype, {2});
    auto *node = graph->add_node();
    node->set_op_type("Cast");
    node->add_input("data");
    node->add_output("output");
    AddAttribute<int64_t>(*node, "to", TensorProto::INT64);
    graph->add_output()->set_name("output");
    ASSERT_NO_THROW(InferNDModel(model));
    EXPECT_THROW(version_conversion::ConvertVersion(model, 27), std::runtime_error);
    graph->mutable_input(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
        TensorProto::FLOAT);
    *node->mutable_output(0) = "intermediate";
    node->mutable_attribute(0)->set_i(dtype);
    auto *cast_back = graph->add_node();
    cast_back->set_op_type("Cast");
    cast_back->add_input("intermediate");
    cast_back->add_output("output");
    AddAttribute<int64_t>(*cast_back, "to", TensorProto::INT64);
    graph->clear_value_info();
    EXPECT_THROW(version_conversion::ConvertVersion(model, 27), std::runtime_error);
  }
}
