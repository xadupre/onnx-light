// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_lib/checker.h"
#include "onnx_lib/version_converter/convert.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace Test {

TEST(MeanVarianceNormalizationConverterExecution, PreservesBothVersionsForAllFloatingTypes) {
  RegisterAllOnnxOperatorSchemas();
  onnx_kernels::RegisterKernelFunctions();
  for (const bool upgrade : {false, true}) {
    for (const auto dtype :
         {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT, TensorProto::DOUBLE}) {
      for (const bool explicit_axes : {false, true}) {
        for (const float epsilon : {1e-9f, 1.0f}) {
          if (upgrade && epsilon != 1e-9f)
            continue;
          for (const std::vector<float> &values :
               {std::vector<float>{1, 3, 5, 7}, {-3, 3, -1, 1}}) {
            SCOPED_TRACE(upgrade);
            SCOPED_TRACE(dtype);
            SCOPED_TRACE(explicit_axes);
            SCOPED_TRACE(epsilon);
            SCOPED_TRACE(values[0]);
            ModelProto model;
            model.set_ir_version(IR_VERSION);
            model.add_opset("", upgrade ? 28 : 29);
            auto *graph = model.mutable_graph();
            graph->set_name("converted_mvn_execution");
            auto *input = graph->add_input();
            input->set_name("X");
            auto *type = input->mutable_type()->mutable_tensor_type();
            type->set_elem_type(dtype);
            for (const int64_t dim : {2, 2, 1, 1})
              type->mutable_shape()->add_dim()->set_dim_value(dim);
            auto *output = graph->add_output();
            output->set_name("Y");
            output->mutable_type()->CopyFrom(input->type());
            auto *node = graph->add_node();
            node->set_op_type("MeanVarianceNormalization");
            node->add_input("X");
            node->add_output("Y");
            if (explicit_axes) {
              auto *axes = node->add_attribute();
              axes->set_name("axes");
              axes->set_type(AttributeProto::INTS);
              axes->add_ints(1);
            }
            if (!upgrade) {
              auto *attribute = node->add_attribute();
              attribute->set_name("epsilon");
              attribute->set_type(AttributeProto::FLOAT);
              attribute->set_f(epsilon);
            }
            const auto converted = version_conversion::ConvertVersion(model, upgrade ? 29 : 28);
            ASSERT_NO_THROW(checker::check_model(converted, true));
            auto x = dtype == TensorProto::DOUBLE
                         ? core::runtime::Tensor::FromDouble(
                               "X", {2, 2, 1, 1}, std::vector<double>(values.begin(), values.end()))
                         : core::runtime::Tensor::FromFloat("X", {2, 2, 1, 1}, values);
            if (core::runtime::IsHalfPrecision(dtype))
              x = core::runtime::DemoteFromFloat32(x, dtype);
            const auto outputs = core::runtime::RunModel(converted, {std::move(x)});
            ASSERT_EQ(outputs.size(), 1u);
            EXPECT_EQ(outputs[0].data_type, dtype);
            EXPECT_EQ(outputs[0].shape, (core::runtime::Shape{2, 2, 1, 1}));
            const auto actual = core::runtime::PromoteToFloat32(outputs[0]);
            for (size_t index = 0; index < values.size(); ++index) {
              const double other = values[index ^ (explicit_axes ? 1u : 2u)];
              const double difference = static_cast<double>(values[index]) - other;
              const double expected =
                  difference / 2 / (std::abs(difference) / 2 + static_cast<double>(epsilon));
              const double value =
                  dtype == TensorProto::DOUBLE ? actual.AsDouble()[index] : actual.AsFloat()[index];
              EXPECT_TRUE(std::isfinite(value));
              const double tolerance = dtype == TensorProto::DOUBLE  ? 1e-12
                                       : dtype == TensorProto::FLOAT ? 1e-6
                                                                     : 8e-3;
              EXPECT_NEAR(value, expected, tolerance);
            }
          }
        }
      }
    }
  }
}

TEST(MeanVarianceNormalizationConverterExecution, PreservesLegacyAndCenteredVarianceAtLargeOffset) {
  RegisterAllOnnxOperatorSchemas();
  onnx_kernels::RegisterKernelFunctions();
  const std::vector<float> values{10000, 10100};
  for (const bool upgrade : {false, true}) {
    SCOPED_TRACE(upgrade);
    ModelProto model;
    model.set_ir_version(IR_VERSION);
    model.add_opset("", upgrade ? 28 : 29);
    auto *graph = model.mutable_graph();
    graph->set_name("converted_mvn_large_offset");
    auto *input = graph->add_input();
    input->set_name("X");
    auto *type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(2);
    auto *output = graph->add_output();
    output->set_name("Y");
    output->mutable_type()->CopyFrom(input->type());
    auto *node = graph->add_node();
    node->set_op_type("MeanVarianceNormalization");
    node->add_input("X");
    node->add_output("Y");
    auto *axes = node->add_attribute();
    axes->set_name("axes");
    axes->set_type(AttributeProto::INTS);
    axes->add_ints(0);
    const auto converted = version_conversion::ConvertVersion(model, upgrade ? 29 : 28);
    ASSERT_NO_THROW(checker::check_model(converted, true));
    const auto outputs =
        core::runtime::RunModel(converted, {core::runtime::Tensor::FromFloat("X", {2}, values)});
    ASSERT_EQ(outputs.size(), 1u);
    const float mean = (values[0] + values[1]) / 2;
    const float first_centered = values[0] - mean;
    const float second_centered = values[1] - mean;
    const float variance =
        upgrade ? (values[0] * values[0] + values[1] * values[1]) / 2 - mean * mean
                : (first_centered * first_centered + second_centered * second_centered) / 2;
    for (size_t index = 0; index < values.size(); ++index) {
      const float expected = (values[index] - mean) / (std::sqrt(variance) + 1e-9f);
      EXPECT_TRUE(std::isfinite(outputs[0].AsFloat()[index]));
      EXPECT_NEAR(outputs[0].AsFloat()[index], expected, std::abs(expected) * 1e-6f + 1e-6f);
    }
  }
}

} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
