// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/defs/operator_sets.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

TEST(LogSoftmaxFunction, KeepsVersionedDependenciesAndPromotesHalfInputs) {
  RegisterOnnxOperatorSetSchema(0, false);
  for (const int version : {13, 18, 28, 29}) {
    const auto *schema = OpSchemaRegistry::Schema("LogSoftmax", version, ONNX_DOMAIN);
    ASSERT_NE(schema, nullptr);
    EXPECT_EQ(schema->SinceVersion(), 13);
    EXPECT_EQ(schema->context_dependent_function_opset_versions(), (std::vector<int>{13, 18, 29}));
    for (const auto dtype :
         {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
      for (const int axis : {-1, 0, 1}) {
        NodeProto node;
        node.set_op_type("LogSoftmax");
        node.add_input("x");
        node.add_output("y");
        if (axis != -1) {
          auto *attribute = node.add_attribute();
          attribute->set_name("axis");
          attribute->set_type(AttributeProto::INT);
          attribute->set_i(axis);
        }
        TypeProto input_type;
        input_type.mutable_tensor_type()->set_elem_type(dtype);
        FunctionBodyBuildContextImpl context(node, std::vector<TypeProto>{input_type});
        FunctionProto function;
        ASSERT_TRUE(schema->BuildContextDependentFunction(context, function, version));
        ASSERT_EQ(function.opset_import_size(), 1);
        EXPECT_EQ(function.opset_import(0).version(), version);
        bool found_log1p = false;
        bool found_axes = false;
        for (const auto &internal : function.node()) {
          found_log1p |= internal.op_type() == "Log1p";
          EXPECT_NE(internal.op_type(), "If");
          EXPECT_NE(internal.op_type(), "ArgMax");
          if (internal.output(0) == "axes") {
            ASSERT_EQ(internal.attribute_size(), 1);
            ASSERT_EQ(internal.attribute(0).ints_size(), 1);
            EXPECT_EQ(internal.attribute(0).ints(0), axis);
            found_axes = true;
          }
          if (version == 29 && internal.output(0) == "X") {
            const bool promote = dtype == TensorProto::FLOAT16 || dtype == TensorProto::BFLOAT16;
            EXPECT_EQ(internal.op_type(), promote ? "Cast" : "Identity");
            if (promote) {
              ASSERT_EQ(internal.attribute_size(), 1);
              EXPECT_EQ(internal.attribute(0).i(), TensorProto::FLOAT);
            }
          }
        }
        EXPECT_EQ(found_log1p, version == 29);
        EXPECT_TRUE(found_axes);
        checker::CheckerContext checker_context;
        checker_context.set_ir_version(IR_VERSION);
        checker_context.set_opset_imports({{ONNX_DOMAIN, version}});
        checker::LexicalScopeContext lexical_scope;
        EXPECT_NO_THROW(checker::check_function(function, checker_context, lexical_scope));
      }
    }
  }
}

TEST(LogSoftmaxFunction, RejectsMissingInputTypeAtOpset29) {
  RegisterOnnxOperatorSetSchema(0, false);
  const auto *schema = OpSchemaRegistry::Schema("LogSoftmax", 29, ONNX_DOMAIN);
  ASSERT_NE(schema, nullptr);
  NodeProto node;
  node.set_op_type("LogSoftmax");
  node.add_input("x");
  node.add_output("y");
  FunctionBodyBuildContextImpl context(node, std::vector<TypeProto>{});
  FunctionProto function;
  EXPECT_FALSE(schema->BuildContextDependentFunction(context, function, 29));
}
