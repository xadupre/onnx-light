// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/checker.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {
ModelProto MakeExpm1Model(TensorProto::DataType data_type, const TensorShapeProto &shape) {
  RegisterAllOnnxOperatorSchemas();
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(29);
  auto *graph = model.mutable_graph();
  graph->set_name("expm1");
  auto *input = graph->add_input();
  input->set_name("x");
  auto *type = input->mutable_type()->mutable_tensor_type();
  type->set_elem_type(data_type);
  *type->mutable_shape() = shape;
  graph->add_output()->set_name("y");
  auto *node = graph->add_node();
  node->set_op_type("Expm1");
  node->add_input("x");
  node->add_output("y");
  return model;
}
} // namespace

TEST(onnx_expm1, InfersTypeAndShapeAndChecksModel) {
  TensorShapeProto scalar;
  TensorShapeProto empty;
  empty.add_dim()->set_dim_value(0);
  empty.add_dim()->set_dim_value(3);
  TensorShapeProto matrix;
  matrix.add_dim()->set_dim_value(2);
  matrix.add_dim()->set_dim_value(3);
  TensorShapeProto symbolic;
  symbolic.add_dim()->set_dim_param("N");
  symbolic.add_dim();
  symbolic.add_dim()->set_dim_value(3);
  for (auto data_type :
       {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT, TensorProto::DOUBLE}) {
    for (const auto &shape : {scalar, empty, matrix, symbolic}) {
      auto model = MakeExpm1Model(data_type, shape);
      shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                                   ShapeInferenceOptions(true, 1, false));
      const auto &output = model.graph().output(0).type().tensor_type();
      EXPECT_EQ(output.elem_type(), data_type);
      ASSERT_EQ(output.shape().dim_size(), shape.dim_size());
      for (int i = 0; i < shape.dim_size(); ++i) {
        const auto &expected = shape.dim(i);
        const auto &actual = output.shape().dim(i);
        if (expected.has_dim_value()) {
          EXPECT_TRUE(actual.has_dim_value());
          EXPECT_EQ(actual.dim_value(), expected.dim_value());
        } else if (expected.has_dim_param()) {
          EXPECT_TRUE(actual.has_dim_param());
          EXPECT_EQ(actual.dim_param(), expected.dim_param());
        } else {
          EXPECT_FALSE(actual.has_dim_value());
        }
      }
      checker::check_model(model, true);
    }
  }
}

TEST(onnx_expm1, RejectsInvalidElementType) {
  auto model = MakeExpm1Model(TensorProto::INT32, TensorShapeProto{});
  EXPECT_THROW(shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                                            ShapeInferenceOptions(true, 1, false)),
               InferenceError);
}

TEST(onnx_expm1, VersionConversionDoesNotDowngradeTo28) {
  auto model = MakeExpm1Model(TensorProto::FLOAT, TensorShapeProto{});
  shape_inference::InferShapes(model);
  const auto unchanged = version_conversion::ConvertVersion(model, 29);
  EXPECT_EQ(unchanged.graph().node(0).op_type(), "Expm1");
  EXPECT_THROW(version_conversion::ConvertVersion(model, 28), std::runtime_error);
}
