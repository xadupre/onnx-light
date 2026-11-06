// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

TEST(GeluPattern, ExactNumericallyMatchesOriginalGraphsAcrossDtypesAndArrangements) {
  onnx_kernels::RegisterKernelFunctions();
  const auto schemas = [](const std::string &op) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false);
  };
  const std::vector<double> values{-std::numeric_limits<double>::infinity(),
                                   -8,
                                   -4,
                                   -3,
                                   -2,
                                   -1,
                                   -0.5,
                                   -0.1,
                                   -0.0,
                                   0.0,
                                   1e-8,
                                   1e-4,
                                   0.1,
                                   0.5,
                                   1,
                                   2,
                                   3,
                                   4,
                                   8,
                                   20,
                                   std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN(),
                                   -1e-4,
                                   -1e-8};
  for (const auto type : {core::symbolic::TensorType::kFloat, core::symbolic::TensorType::kFloat16,
                          core::symbolic::TensorType::kDouble}) {
    const auto dtype =
        static_cast<TensorProto::DataType>(core::symbolic::TensorTypeToDataType(type));
    core::runtime::Tensor input;
    if (type == core::symbolic::TensorType::kDouble) {
      input = core::runtime::Tensor::FromDouble("x", {2, 12}, values);
    } else if (type == core::symbolic::TensorType::kFloat) {
      input = core::runtime::Tensor::FromFloat("x", {2, 12},
                                               std::vector<float>(values.begin(), values.end()));
    } else {
      core::runtime::RawByteBuffer bytes(values.size() * sizeof(uint16_t));
      for (size_t index = 0; index < values.size(); ++index) {
        const uint16_t bits = core::runtime::FloatToFloat16Bits(static_cast<float>(values[index]));
        std::memcpy(bytes.data() + index * sizeof(bits), &bits, sizeof(bits));
      }
      input = core::runtime::Tensor::FromRawBytes("x", dtype, {2, 12}, std::move(bytes));
    }
    const auto element = [type](const core::runtime::Tensor &tensor, size_t index) {
      if (type == core::symbolic::TensorType::kDouble)
        return tensor.AsDouble()[index];
      if (type == core::symbolic::TensorType::kFloat)
        return static_cast<double>(tensor.AsFloat()[index]);
      uint16_t bits;
      std::memcpy(&bits, tensor.bytes() + index * sizeof(bits), sizeof(bits));
      return static_cast<double>(core::runtime::Float16BitsToFloat(bits));
    };
    for (int placement : {0, 1, 2}) {
      for (bool reciprocal : {false, true}) {
        for (bool divide_half : {false, true}) {
          for (bool swapped : {false, true}) {
            SCOPED_TRACE(testing::Message() << static_cast<int>(type) << ":" << placement << ":"
                                            << reciprocal << ":" << divide_half << ":" << swapped);
            core::builder::GraphBuilder builder("exact", schemas, {{"", 20}});
            core::symbolic::SymShape shape;
            for (int64_t dim : {2, 12})
              shape.PushBack(core::symbolic::SymDim(dim));
            builder.MakeInput("x", type, shape);
            for (const auto &[name, value] : std::vector<std::pair<std::string, double>>{
                     {"sqrt_two", reciprocal ? 1.0 / std::sqrt(2.0) : std::sqrt(2.0)},
                     {"one", 1.0},
                     {"half", divide_half ? 2.0 : 0.5}})
              builder.MakeInitializer(
                  onnx_patterns::collections::MakeScalarTensor(name, dtype, {}, value));
            const auto binary = [&](const char *op, std::string left, std::string right,
                                    const std::string &output) {
              if (swapped && std::string(op) != "Div")
                std::swap(left, right);
              builder.MakeNode(op, {left, right}, {output});
            };
            binary(reciprocal ? "Mul" : "Div", "x", "sqrt_two", "scaled");
            builder.MakeNode("Erf", {"scaled"}, {"erf"});
            binary("Add", "erf", "one", "term");
            if (placement == 2) {
              binary("Mul", "x", "term", "product");
              binary(divide_half ? "Div" : "Mul", "product", "half", "y");
            } else {
              binary(divide_half ? "Div" : "Mul", placement == 0 ? "x" : "term", "half", "halved");
              binary("Mul", placement == 0 ? "halved" : "x", placement == 0 ? "term" : "halved",
                     "y");
            }
            builder.MakeOutput("y");
            const auto expected = core::runtime::RunModel(builder.ToModel(), {input});
            std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
            patterns.push_back(onnx_patterns::CreatePattern("Gelu"));
            core::builder::GraphGraph graph(builder, std::move(patterns));
            core::builder::ConstantFoldingOptions folding;
            folding.enabled = false;
            const auto plan = graph.Optimize(4, nullptr, folding);
            ASSERT_FALSE(plan.rewrites().empty());
            EXPECT_EQ(plan.rewrites(0).pattern_name(), "Gelu");
            ASSERT_EQ(builder.Nodes().size(), 1u);
            EXPECT_EQ(builder.Nodes()[0].op_type(), "Gelu");
            EXPECT_EQ(GetAttributeOr<std::string>(builder.Nodes()[0], "approximate", ""), "none");
            const auto actual = core::runtime::RunModel(builder.ToModel(), {input});
            ASSERT_EQ(actual.size(), 1u);
            ASSERT_EQ(expected.size(), 1u);
            EXPECT_EQ(actual[0].shape, expected[0].shape);
            EXPECT_EQ(actual[0].data_type, expected[0].data_type);
            ASSERT_EQ(actual[0].element_count(), static_cast<int64_t>(values.size()));
            const double tolerance =
                type == core::symbolic::TensorType::kFloat16
                    ? 3e-3
                    : (type == core::symbolic::TensorType::kFloat ? 1e-6 : 1e-12);
            for (size_t index = 0; index < values.size(); ++index) {
              const double before = element(expected[0], index);
              const double after = element(actual[0], index);
              if (std::isnan(before))
                EXPECT_TRUE(std::isnan(after)) << index;
              else if (std::isinf(before))
                EXPECT_EQ(after, before) << index;
              else
                EXPECT_NEAR(after, before, tolerance * (1 + std::abs(before))) << index;
            }
          }
        }
      }
    }
  }
}
