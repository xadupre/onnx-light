// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/checker.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

TEST(onnx_log1p, InfersTypeAndShapeAndChecksModel) {
  RegisterAllOnnxOperatorSchemas();
  EXPECT_EQ(OpSchemaRegistry::Schema("Log1p", 28, ONNX_DOMAIN), nullptr);
  const auto *schema = OpSchemaRegistry::Schema("Log1p", 29, ONNX_DOMAIN);
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->SinceVersion(), 29);
  EXPECT_FALSE(schema->HasFunction());
  TensorShapeProto scalar;
  TensorShapeProto empty;
  empty.add_dim()->set_dim_value(0);
  empty.add_dim()->set_dim_value(3);
  TensorShapeProto symbolic;
  symbolic.add_dim()->set_dim_param("N");
  symbolic.add_dim();
  symbolic.add_dim()->set_dim_value(3);
  for (auto data_type : {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT,
                         TensorProto::DOUBLE, TensorProto::INT32}) {
    for (const auto &shape : {scalar, empty, symbolic}) {
      ModelProto model;
      model.set_ir_version(IR_VERSION);
      model.add_opset_import()->set_version(29);
      auto *graph = model.mutable_graph();
      graph->set_name("log1p");
      auto *input = graph->add_input();
      input->set_name("x");
      auto *type = input->mutable_type()->mutable_tensor_type();
      type->set_elem_type(data_type);
      *type->mutable_shape() = shape;
      graph->add_output()->set_name("y");
      auto *node = graph->add_node();
      node->set_op_type("Log1p");
      node->add_input("x");
      node->add_output("y");
      if (data_type == TensorProto::INT32) {
        EXPECT_THROW(shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                                                  ShapeInferenceOptions(true, 1, false)),
                     InferenceError);
        continue;
      }
      shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                                   ShapeInferenceOptions(true, 1, false));
      const auto &output = model.graph().output(0).type().tensor_type();
      EXPECT_EQ(output.elem_type(), data_type);
      EXPECT_EQ(output.shape().SerializeAsString(), shape.SerializeAsString());
      EXPECT_NO_THROW(checker::check_model(model, true));
      const auto unchanged = version_conversion::ConvertVersion(model, 29);
      EXPECT_EQ(unchanged.graph().node(0).op_type(), "Log1p");
      EXPECT_THROW(version_conversion::ConvertVersion(model, 28), std::runtime_error);
    }
  }
}
