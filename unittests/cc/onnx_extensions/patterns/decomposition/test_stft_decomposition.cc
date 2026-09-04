// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/decomposition/stft_decomposition.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape(const std::vector<int64_t> &dims) {
  core::symbolic::SymShape shape;
  for (int64_t dim : dims) {
    shape.PushBack(core::symbolic::SymDim(dim));
  }
  return shape;
}

void AddInt64(core::builder::GraphBuilder &builder, const std::string &name, int64_t value) {
  builder.MakeInitializer(MakeInitializer<int64_t>(name.c_str(), {}, {value}));
}

TEST(STFTDecompositionPattern, BuildsPortableFullSpectrumSubgraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("signal", core::symbolic::TensorType::kDouble, Shape({2, 12, 1}));
  AddInt64(builder, "step", 3);
  AddInt64(builder, "length", 5);
  builder.MakeInitializer(MakeInitializer<double>("window", {5}, {0.2, 0.5, 1.0, 0.5, 0.2}));
  NodeProto stft = MakeNode("STFT", {"signal", "step", "window", "length"}, {"output"});
  AddAttribute<int64_t>(stft, "onesided", 0);
  builder.MakeNode("STFT", {"signal", "step", "window", "length"}, {"output"}, "", "",
                   stft.attribute());
  builder.MakeOutput("output");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::STFTDecompositionPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);

  ASSERT_EQ(replacements.size(), 7u);
  EXPECT_EQ(replacements[0].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[1].op_type().value(), "Conv");
  EXPECT_EQ(replacements[2].op_type().value(), "Conv");
  EXPECT_EQ(replacements[3].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[4].op_type().value(), "Unsqueeze");
  EXPECT_EQ(replacements[5].op_type().value(), "Concat");
  EXPECT_EQ(replacements[6].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[6].output()[0].value(), "output");
  ASSERT_EQ(builder.Initializers().size(), 6u);
  EXPECT_EQ(builder.Initializers()[3].data_type(), TensorProto::DataType::DOUBLE);
  EXPECT_EQ(builder.Initializers()[3].dims(0), 5);
  EXPECT_EQ(builder.Initializers()[3].dims(2), 5);
}

TEST(STFTDecompositionPattern, RejectsDynamicAndUnsupportedInputs) {
  struct TestCase {
    bool dynamic_step;
    bool dynamic_window;
    bool dynamic_length;
    core::symbolic::TensorType signal_type;
    int64_t components;
  };
  const std::vector<TestCase> cases = {
      {true, false, false, core::symbolic::TensorType::kFloat, 1},
      {false, true, false, core::symbolic::TensorType::kFloat, 1},
      {false, false, true, core::symbolic::TensorType::kFloat, 1},
      {false, false, false, core::symbolic::TensorType::kFloat16, 1},
      {false, false, false, core::symbolic::TensorType::kFloat, 2},
  };

  for (const TestCase &test : cases) {
    core::builder::GraphBuilder builder("g", SchemaLookup());
    builder.MakeInput("signal", test.signal_type, Shape({1, 12, test.components}));
    if (test.dynamic_step) {
      builder.MakeInput("step", core::symbolic::TensorType::kInt64, Shape({}));
    } else {
      AddInt64(builder, "step", 2);
    }
    if (test.dynamic_window) {
      builder.MakeInput("window", test.signal_type, Shape({4}));
    } else if (test.signal_type == core::symbolic::TensorType::kFloat) {
      builder.MakeInitializer(MakeInitializer<float>("window", {4}, {1.0F, 1.0F, 1.0F, 1.0F}));
    } else {
      builder.MakeInitializer(MakeInitializer<uint16_t>("window", {4}, {1, 1, 1, 1}));
    }
    if (test.dynamic_length) {
      builder.MakeInput("length", core::symbolic::TensorType::kInt64, Shape({}));
    } else {
      AddInt64(builder, "length", 4);
    }
    NodeProto stft = MakeNode("STFT", {"signal", "step", "window", "length"}, {"output"});
    if (test.components == 2) {
      AddAttribute<int64_t>(stft, "onesided", 0);
    }
    builder.MakeNode("STFT", {"signal", "step", "window", "length"}, {"output"}, "", "",
                     stft.attribute());
    builder.MakeOutput("output");

    core::builder::GraphGraph graph(builder);
    onnx_patterns::STFTDecompositionPattern pattern;
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
  }
}

TEST(STFTDecompositionPattern, RejectsInvalidConstantsAndShortSignal) {
  for (const std::pair<int64_t, int64_t> &values :
       {std::pair<int64_t, int64_t>{0, 4}, {2, 0}, {2, -1}, {2, 4}}) {
    core::builder::GraphBuilder builder("g", SchemaLookup());
    builder.MakeInput("signal", core::symbolic::TensorType::kFloat, Shape({1, 3, 1}));
    AddInt64(builder, "step", values.first);
    AddInt64(builder, "length", values.second);
    builder.MakeNode("STFT", {"signal", "step", "", "length"}, {"output"});
    builder.MakeOutput("output");

    core::builder::GraphGraph graph(builder);
    onnx_patterns::STFTDecompositionPattern pattern;
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
  }
}

TEST(STFTDecompositionPattern, RejectsMismatchedWindow) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("signal", core::symbolic::TensorType::kFloat, Shape({1, 12, 1}));
  AddInt64(builder, "step", 2);
  AddInt64(builder, "length", 5);
  builder.MakeInitializer(MakeInitializer<float>("window", {4}, {1.0F, 1.0F, 1.0F, 1.0F}));
  builder.MakeNode("STFT", {"signal", "step", "window", "length"}, {"output"});
  builder.MakeOutput("output");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::STFTDecompositionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

} // namespace
} // namespace Test
