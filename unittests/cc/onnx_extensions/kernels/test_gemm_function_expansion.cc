// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_lib/defs/function.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

TEST(Gemm29, Float16FunctionExpansionMatchesNativeExecution) {
  onnx_light::RegisterAllOnnxOperatorSchemas();
  onnx_light::ModelProto model;
  model.set_ir_version(onnx_light::IR_VERSION);
  model.add_opset_import()->set_version(29);
  onnx_light::GraphProto *graph = model.mutable_graph();
  graph->set_name("gemm29_float16");
  for (const auto &name : {"A", "B"}) {
    onnx_light::ValueInfoProto *input = graph->add_input();
    input->set_name(name);
    auto *tensor_type = input->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(onnx_light::TensorProto::FLOAT16);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
    if (std::string_view(name) == "B") {
      tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
    }
  }
  onnx_light::ValueInfoProto *output = graph->add_output();
  output->set_name("Y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(onnx_light::TensorProto::FLOAT16);
  *graph->add_node() = onnx_light::MakeNode("Gemm", {"A", "B"}, {"Y"});
  onnx_light::NodeProto node = graph->node(0);
  onnx_light::AddAttribute<float>(node, "alpha", 0.5F);
  *graph->mutable_node(0) = node;

  std::vector<onnx_light::TypeProto> input_types;
  for (const onnx_light::ValueInfoProto &input : graph->input()) {
    input_types.push_back(input.type());
  }
  const auto *schema = onnx_light::OpSchemaRegistry::Schema("Gemm", 29, "");
  ASSERT_NE(schema, nullptr);
  onnx_light::FunctionBodyBuildContextImpl context(node, input_types);
  onnx_light::FunctionProto function;
  ASSERT_TRUE(schema->BuildContextDependentFunction(context, function, 29));
  onnx_light::FunctionExpandHelper(node, function, *graph);

  const onnx_light::core::runtime::Tensor a =
      onnx_light::core::runtime::MakeFloat16Tensor("A", {1}, {60000.0F});
  const onnx_light::core::runtime::Tensor b =
      onnx_light::core::runtime::MakeFloat16Tensor("B", {1, 1}, {2.0F});
  const onnx_light::core::runtime::Tensor native =
      onnx_light::onnx_kernels::kernel::Gemm{onnx_light::onnx_kernels::kernel::KernelContext{
          onnx_light::core::runtime::DefaultOpset(29)}}(a, b, nullptr, 0.5F, 1.0F, 0, 0);
  const auto expanded = onnx_light::core::runtime::RunModel(model, {a, b});
  ASSERT_EQ(expanded.size(), 1u);
  ASSERT_EQ(expanded[0].data_type, onnx_light::TensorProto::FLOAT16);
  const std::uint16_t expected = onnx_light::core::runtime::FloatToFloat16Bits(60000.0F);
  EXPECT_EQ(std::memcmp(native.bytes(), &expected, sizeof(expected)), 0);
  EXPECT_EQ(std::memcmp(expanded[0].bytes(), native.bytes(), native.size_bytes()), 0);
}
