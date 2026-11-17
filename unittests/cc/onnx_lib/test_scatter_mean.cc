// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_lib/version_converter/convert.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto MakeScatterModel(const std::string &op, int dtype, int version,
                            const std::string &reduction, int index_type = TensorProto::INT64,
                            bool has_shape = true) {
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  model.add_opset_import()->set_version(version);
  auto *graph = model.mutable_graph();
  graph->set_name("scatter_mean");
  for (const std::string name : {"data", "indices", "updates"}) {
    auto *input = graph->add_input();
    input->set_name(name);
    auto *type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(name == "indices" ? index_type : dtype);
    if (has_shape) {
      auto *shape = type->mutable_shape();
      if (name == "data") {
        shape->add_dim()->set_dim_param("N");
        shape->add_dim()->set_dim_value(3);
      } else {
        shape->add_dim()->set_dim_value(2);
        if (op == "ScatterElements" || name == "indices") {
          shape->add_dim()->set_dim_value(2);
        }
      }
    }
  }
  auto *node = graph->add_node();
  node->set_op_type(op);
  for (const std::string input : {"data", "indices", "updates"}) {
    node->add_input(input);
  }
  node->add_output("output");
  if (!reduction.empty()) {
    AddAttribute<std::string>(*node, "reduction", reduction);
  }
  graph->add_output()->set_name("output");
  return model;
}

void InferScatter(ModelProto &model) {
  shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                               ShapeInferenceOptions(true, 1, false));
}

} // namespace

TEST(ScatterMean, PreservesHistoricalSchemas) {
  RegisterAllOnnxOperatorSchemas();
  for (const std::string op : {"ScatterElements", "ScatterND"}) {
    for (int version : {18, 28, 29}) {
      const auto *schema = OpSchemaRegistry::Schema(op, version, ONNX_DOMAIN);
      ASSERT_NE(schema, nullptr);
      EXPECT_EQ(schema->SinceVersion(), version == 29 ? 29 : 18);
      const auto &description = schema->attributes().at("reduction").description;
      EXPECT_EQ(description.find("'mean'") != std::string::npos, version == 29);
    }
  }
}

TEST(ScatterMean, InfersSymbolicShapesAndNumericTypes) {
  RegisterAllOnnxOperatorSchemas();
  for (const std::string op : {"ScatterElements", "ScatterND"}) {
    for (int dtype :
         {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16,
          TensorProto::INT32, TensorProto::INT64, TensorProto::UINT64}) {
      for (int index_type : {TensorProto::INT32, TensorProto::INT64}) {
        auto model = MakeScatterModel(op, dtype, 29, "mean", index_type);
        ASSERT_NO_THROW(InferScatter(model));
        const auto &output = model.graph().output(0).type().tensor_type();
        EXPECT_EQ(output.elem_type(), dtype);
        EXPECT_EQ(output.shape().SerializeAsString(),
                  model.graph().input(0).type().tensor_type().shape().SerializeAsString());
        EXPECT_NO_THROW(checker::check_model(model, true));
      }
    }
  }
}

TEST(ScatterMean, RejectsInvalidReductionAndUnsupportedTypesWithoutShapes) {
  RegisterAllOnnxOperatorSchemas();
  for (const std::string op : {"ScatterElements", "ScatterND"}) {
    for (bool has_shape : {false, true}) {
      auto invalid =
          MakeScatterModel(op, TensorProto::FLOAT, 29, "average", TensorProto::INT64, has_shape);
      EXPECT_THROW(InferScatter(invalid), InferenceError);
      for (int dtype : {TensorProto::BOOL, TensorProto::STRING, TensorProto::COMPLEX64,
                        TensorProto::COMPLEX128}) {
        auto model = MakeScatterModel(op, dtype, 29, "mean", TensorProto::INT64, has_shape);
        EXPECT_THROW(InferScatter(model), InferenceError);
        auto unchanged = MakeScatterModel(op, dtype, 29, "none", TensorProto::INT64, has_shape);
        EXPECT_NO_THROW(InferScatter(unchanged));
      }
    }
  }
}

TEST(ScatterMean, PreservesEmptyShapesAndInfersTypesWithoutInputShapes) {
  RegisterAllOnnxOperatorSchemas();
  for (const std::string op : {"ScatterElements", "ScatterND"}) {
    auto unknown = MakeScatterModel(op, TensorProto::INT64, 29, "mean", TensorProto::INT64, false);
    ASSERT_NO_THROW(InferScatter(unknown));
    const auto &unknown_output = unknown.graph().output(0).type().tensor_type();
    EXPECT_EQ(unknown_output.elem_type(), TensorProto::INT64);
    EXPECT_FALSE(unknown_output.has_shape());

    auto empty = MakeScatterModel(op, TensorProto::FLOAT, 29, "mean");
    for (int input : {0, 1, 2}) {
      empty.mutable_graph()
          ->mutable_input(input)
          ->mutable_type()
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(0)
          ->set_dim_value(0);
    }
    ASSERT_NO_THROW(InferScatter(empty));
    EXPECT_EQ(empty.graph().output(0).type().tensor_type().shape().dim(0).dim_value(), 0);

    if (op == "ScatterND") {
      auto whole_tensor = MakeScatterModel(op, TensorProto::FLOAT, 29, "mean");
      auto *graph = whole_tensor.mutable_graph();
      graph->mutable_input(1)
          ->mutable_type()
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_dim(1)
          ->set_dim_value(0);
      auto *updates_shape =
          graph->mutable_input(2)->mutable_type()->mutable_tensor_type()->mutable_shape();
      updates_shape->add_dim()->set_dim_param("N");
      updates_shape->add_dim()->set_dim_value(3);
      ASSERT_NO_THROW(InferScatter(whole_tensor));
      EXPECT_EQ(whole_tensor.graph().output(0).type().tensor_type().shape().SerializeAsString(),
                whole_tensor.graph().input(0).type().tensor_type().shape().SerializeAsString());
    }
  }
}

TEST(ScatterMean, ConverterRejectsMeanAndPreservesLegacyReductions) {
  RegisterAllOnnxOperatorSchemas();
  for (const std::string op : {"ScatterElements", "ScatterND"}) {
    for (int index_type : {TensorProto::INT32, TensorProto::INT64}) {
      auto mean = MakeScatterModel(op, TensorProto::FLOAT, 29, "mean", index_type);
      ASSERT_NO_THROW(InferScatter(mean));
      EXPECT_THROW(version_conversion::ConvertVersion(mean, 28), std::runtime_error);
      for (const std::string reduction : {"", "none", "add", "mul", "max", "min"}) {
        auto model = MakeScatterModel(op, TensorProto::FLOAT, 29, reduction, index_type);
        ASSERT_NO_THROW(InferScatter(model));
        auto converted = version_conversion::ConvertVersion(model, 28);
        ASSERT_GT(converted.graph().node_size(), 0);
        const auto &node = converted.graph().node(converted.graph().node_size() - 1);
        EXPECT_EQ(node.op_type(), op);
        EXPECT_EQ(node.attribute_size(), model.graph().node(0).attribute_size());
        if (!reduction.empty()) {
          EXPECT_EQ(node.attribute(0).s(), reduction);
        }
        EXPECT_EQ(converted.graph().input(1).type().tensor_type().elem_type(), index_type);
        const bool casts_indices = op == "ScatterND" && index_type == TensorProto::INT32;
        EXPECT_EQ(converted.graph().node_size(), casts_indices ? 2 : 1);
        EXPECT_NO_THROW(InferScatter(converted));
        EXPECT_NO_THROW(checker::check_model(converted, true));
        auto upgraded = version_conversion::ConvertVersion(converted, 29);
        EXPECT_EQ(upgraded.opset_import(0).version(), 29);
        EXPECT_NO_THROW(InferScatter(upgraded));
      }
    }
  }
}

TEST(ScatterMean, ExtensionShapesValidateOpset29WithoutChangingLegacyBehavior) {
  using namespace onnx_shapes::shapes;
  for (const std::string op : {"ScatterElements", "ScatterND"}) {
    for (int version : {kUnknownOpsetVersion, 28, 29}) {
      for (int dtype : {TensorProto::FLOAT, TensorProto::BOOL, TensorProto::STRING,
                        TensorProto::COMPLEX64, TensorProto::COMPLEX128}) {
        for (const std::string reduction : {"mean", "average", "none"}) {
          ShapesContext ctx;
          if (version != kUnknownOpsetVersion) {
            ctx.SetOpsetVersion("", version);
          }
          const auto tensor_type =
              core::symbolic::DataTypeToTensorType(static_cast<TensorProto::DataType>(dtype));
          ctx.Set("data", SymTensor(nullptr, tensor_type, SymShape({2, 3})));
          auto model = MakeScatterModel(op, dtype, 29, reduction);
          const auto &node = model.graph().node(0);
          auto infer = [&]() {
            if (op == "ScatterElements") {
              tensor::ComputeShapeScatterElements(ctx, node);
            } else {
              tensor::ComputeShapeScatterND(ctx, node);
            }
          };
          if (version != 28 &&
              (reduction == "average" || (reduction == "mean" && dtype != TensorProto::FLOAT))) {
            EXPECT_THROW(infer(), std::invalid_argument);
          } else {
            ASSERT_NO_THROW(infer());
            EXPECT_EQ(ctx.Get("output").Dtype(), tensor_type);
            EXPECT_EQ(ctx.Get("output").Shape().Rank(), 2u);
          }
        }
      }
    }
  }
}
