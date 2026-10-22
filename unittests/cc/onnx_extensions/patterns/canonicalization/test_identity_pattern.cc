// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/canonicalization/identity_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape() {
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(3));
  return shape;
}

TEST(IdentityPattern, MatchesAddZeroScalar) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("zero", {1}, {0.0f}));
  builder.MakeNode("Add", {"x", "zero"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::IdentityPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
}

TEST(IdentityPattern, MatchesMulOneScalarOnLeft) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("one", {1}, {1.0f}));
  builder.MakeNode("Mul", {"one", "x"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::IdentityPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
}

TEST(IdentityPattern, MatchesIdentityTranspose) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(2));
  shape.PushBack(core::symbolic::SymDim(3));
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &perm = attributes.add();
  perm.set_name("perm");
  perm.set_type(AttributeProto::AttributeType::INTS);
  perm.ref_ints().push_back(0);
  perm.ref_ints().push_back(1);
  builder.MakeNode("Transpose", {"x"}, {"y"}, "", "", attributes);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::IdentityPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);
}

TEST(IdentityPattern, OptimizeReplacesNoOpAdd) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("zero", {1}, {0.0f}));
  builder.MakeNode("Add", {"x", "zero"}, {"y"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::IdentityPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Identity");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "y");
}

template <typename T> void CheckConstantOperands(T value) {
  for (const std::string op_type : {"Add", "Sub", "Mul", "Div"}) {
    SCOPED_TRACE(op_type);
    const T neutral = (op_type == "Add" || op_type == "Sub") ? T(0) : T(1);
    for (const std::vector<int64_t> &dims : {std::vector<int64_t>{}, std::vector<int64_t>{1}}) {
      SCOPED_TRACE(dims.size());
      for (bool neutral_on_left : {false, true}) {
        SCOPED_TRACE(neutral_on_left);
        core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
        TensorProto data = MakeInitializer<T>("data", dims, {value});
        const TensorProto identity = MakeInitializer<T>("neutral", dims, {neutral});
        builder.MakeInitializer(data);
        builder.MakeInitializer(identity);
        const std::vector<std::string> inputs = neutral_on_left
                                                    ? std::vector<std::string>{"neutral", "data"}
                                                    : std::vector<std::string>{"data", "neutral"};
        builder.MakeNode(op_type, inputs, {"y"});
        builder.MakeOutput("y");
        const NodeProto original = builder.Nodes()[0];

        std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
        patterns.push_back(std::make_unique<onnx_patterns::IdentityPattern>());
        core::builder::GraphGraph graph(builder, std::move(patterns));
        onnx_patterns::IdentityPattern pattern;
        const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
        if (neutral_on_left) {
          // A non-neutral constant on the right prevents a match, even for Add/Mul.
          EXPECT_EQ(match.pattern, nullptr);
        } else {
          ASSERT_EQ(match.pattern, &pattern);
          const auto replacements = pattern.Apply(graph, match.nodes);
          ASSERT_EQ(replacements.size(), 1u);
          EXPECT_EQ(replacements[0].op_type().value(), "Identity");
          EXPECT_EQ(replacements[0].input()[0].value(), "data");
          EXPECT_EQ(replacements[0].output()[0].value(), "y");
        }

        graph.Optimize();
        ModelProto optimized = builder.ToModel();
        for (TensorProto &initializer : optimized.ref_graph().ref_initializer()) {
          initializer.ref_metadata_props().clear();
        }
        for (NodeProto &node : optimized.ref_graph().ref_node()) {
          node.ref_metadata_props().clear();
        }
        ASSERT_EQ(optimized.graph().output_size(), 1);
        EXPECT_EQ(optimized.graph().output()[0].name().value(), "y");
        if (neutral_on_left) {
          ASSERT_EQ(optimized.graph().node_size(), 1);
          EXPECT_EQ(optimized.graph().node()[0].SerializeAsString(), original.SerializeAsString());
          ASSERT_EQ(optimized.graph().initializer_size(), 2);
          EXPECT_EQ(optimized.graph().initializer()[0].SerializeAsString(),
                    data.SerializeAsString());
          EXPECT_EQ(optimized.graph().initializer()[1].SerializeAsString(),
                    identity.SerializeAsString());
        } else {
          if (optimized.graph().node_size() == 0) {
            data.set_name("y");
          } else {
            ASSERT_EQ(optimized.graph().node_size(), 1);
            const NodeProto &node = optimized.graph().node()[0];
            EXPECT_EQ(node.op_type().value(), "Identity");
            ASSERT_EQ(node.input_size(), 1);
            EXPECT_EQ(node.input()[0].value(), "data");
            ASSERT_EQ(node.output_size(), 1);
            EXPECT_EQ(node.output()[0].value(), "y");
          }
          ASSERT_EQ(optimized.graph().initializer_size(), 1);
          const TensorProto &result = optimized.graph().initializer()[0];
          EXPECT_EQ(result.name().value(), data.name().value());
          EXPECT_EQ(result.data_type(), data.data_type());
          ASSERT_EQ(result.dims_size(), data.dims_size());
          for (int i = 0; i < result.dims_size(); ++i) {
            EXPECT_EQ(result.dims()[i], data.dims()[i]);
          }
          if constexpr (std::is_floating_point_v<T>) {
            std::vector<double> values;
            ASSERT_TRUE(ReadFloatingValues(result, values));
            EXPECT_EQ(values, std::vector<double>{static_cast<double>(value)});
          } else {
            std::vector<int64_t> values;
            ASSERT_TRUE(ReadIntegerValues(result, values));
            EXPECT_EQ(values, std::vector<int64_t>{static_cast<int64_t>(value)});
          }
        }
      }
    }
  }
}

TEST(IdentityPattern, PreservesFloatConstantOperands) {
  for (float value : {0.5f, 1.5f, -0.5f}) {
    SCOPED_TRACE(value);
    CheckConstantOperands(value);
  }
}

TEST(IdentityPattern, PreservesDoubleConstantOperands) {
  for (double value : {0.5, 1.0000000000000002, -0.5}) {
    SCOPED_TRACE(value);
    CheckConstantOperands(value);
  }
}

TEST(IdentityPattern, PreservesInt32ConstantOperands) { CheckConstantOperands<int32_t>(7); }

TEST(IdentityPattern, PreservesInt64ConstantOperands) {
  CheckConstantOperands<int64_t>(9007199254740993LL);
}

TEST(IdentityPattern, RejectsNonNeutralScalar) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("five", {1}, {5.0f}));
  builder.MakeNode("Add", {"x", "five"}, {"y"});
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::IdentityPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the scalar constant operand is not a neutral element");
}

template <typename T> void CheckNeutralVectors() {
  for (const std::string op_type : {"Add", "Sub", "Mul", "Div"}) {
    SCOPED_TRACE(op_type);
    const T neutral = (op_type == "Add" || op_type == "Sub") ? T(0) : T(1);
    std::vector<std::vector<T>> candidates = {
        {neutral, neutral, neutral},
        {neutral, T(2), neutral},
        {neutral, neutral},
        {neutral, neutral, neutral, neutral},
    };
    if constexpr (std::is_floating_point_v<T>)
      candidates.push_back({neutral, std::numeric_limits<T>::quiet_NaN(), neutral});
    for (const auto &values : candidates) {
      SCOPED_TRACE(values.size());
      auto constant = MakeInitializer<T>("neutral", {3}, values);
      constant.set_raw_data(std::string(3 * sizeof(T), '\0'));
      core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
      builder.MakeInput("x", core::symbolic::DataTypeToTensorType(constant.data_type()), Shape());
      builder.MakeInitializer(std::move(constant));
      builder.MakeNode(op_type, {"x", "neutral"}, {"y"});
      builder.MakeOutput("y");
      core::builder::GraphGraph graph(builder);
      onnx_patterns::IdentityPattern pattern;
      const auto match = pattern.Match(graph, builder.Nodes()[0]);
      if (values.size() == 3 && values[1] == neutral) {
        ASSERT_EQ(match.pattern, &pattern);
        const auto replacements = pattern.Apply(graph, match.nodes);
        ASSERT_EQ(replacements.size(), 1u);
        EXPECT_EQ(replacements[0].op_type().value(), "Identity");
        EXPECT_EQ(replacements[0].input()[0].value(), "x");
        EXPECT_EQ(replacements[0].output()[0].value(), "y");
      } else {
        EXPECT_EQ(match.pattern, nullptr);
      }
    }
  }
}

TEST(IdentityPattern, NeutralVectorScanValidatesEveryValueAndPayloadCount) {
  CheckNeutralVectors<float>();
  CheckNeutralVectors<double>();
  CheckNeutralVectors<int32_t>();
  CheckNeutralVectors<int64_t>();
}

} // namespace
} // namespace Test
