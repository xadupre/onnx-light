// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/patterns/canonicalization/cast_pattern.h"
#include "onnx_extensions/patterns/canonicalization/clip_pattern.h"
#include "onnx_extensions/patterns/canonicalization/constant_pattern.h"
#include "onnx_extensions/patterns/canonicalization/conv_pattern.h"
#include "onnx_extensions/patterns/canonicalization/dropout_pattern.h"
#include "onnx_extensions/patterns/canonicalization/identity_pattern.h"
#include "onnx_extensions/patterns/canonicalization/not_pattern.h"
#include "onnx_extensions/patterns/dispatch_table.h"

#include "onnx_helper.h"
#include "onnx_op/operator_sets.h"

#include <algorithm>
#include <memory>

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
  shape.PushBack(core::symbolic::SymDim(2));
  return shape;
}

void AddCast(core::builder::GraphBuilder &builder, const std::string &input,
             const std::string &output, TensorProto::DataType to, const std::string &name = "") {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("to");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(static_cast<int64_t>(to));
  builder.MakeNode("Cast", {input}, {output}, "", name, attributes);
}

void AddSubgraphReference(core::builder::GraphBuilder &builder, const std::string &attribute_name,
                          const std::string &subgraph_name, const std::string &output) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name(attribute_name + "_ref");
  attribute.set_type(AttributeProto::AttributeType::STRING);
  attribute.set_s(subgraph_name);
  builder.MakeNode("SubgraphCarrier", {}, {output}, "", "", attributes);
}

class ReplaceIdentityWithConstantAddPattern final : public core::builder::PatternOptimization {
public:
  ReplaceIdentityWithConstantAddPattern()
      : PatternOptimization(1, "ReplaceIdentityWithConstantAdd") {}

  std::set<std::string> FastOpType() const override { return {"Identity"}; }

  core::builder::MatchResult Match(core::builder::GraphGraph &graph,
                                   const NodeProto &candidate) const override {
    if (candidate.op_type().value() != "Identity" || candidate.output_size() != 1 ||
        !graph.IsConstant("a") || !graph.IsConstant("b")) {
      return {};
    }
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &nodes) const override {
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(MakeNode("Add", {"a", "b"}, {nodes[0]->output()[0].value()}));
    return replacements;
  }
};

class CountingFastOpTypePattern final : public core::builder::PatternOptimization {
public:
  CountingFastOpTypePattern() : PatternOptimization(1, "CountingFastOpType") {}

  std::set<std::string> FastOpType() const override {
    ++fast_op_type_calls;
    return {"Add"};
  }

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &nodes) const override {
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(*nodes[0]);
    return replacements;
  }

  mutable std::size_t fast_op_type_calls = 0;
};

class NullPlaceholderPattern final : public core::builder::PatternOptimization {
public:
  NullPlaceholderPattern() : PatternOptimization(1, "NullPlaceholder") {}

  std::set<std::string> FastOpType() const override { return {"Identity"}; }

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    return core::builder::MatchResult{this, {nullptr, &candidate}, &candidate};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &nodes) const override {
    if (nodes.size() != 2 || nodes[0] != nullptr || nodes[1] == nullptr) {
      throw core::builder::BuilderError("NullPlaceholderPattern received invalid role slots.");
    }
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(
        MakeNode("Relu", {nodes[1]->input()[0].value()}, {nodes[1]->output()[0].value()}));
    return replacements;
  }
};

class ScopedAddKernel {
public:
  ScopedAddKernel() {
    const auto existing = core::runtime::GlobalCustomKernels().find(kDefaultOnnxDomain + ":Add");
    if (existing != core::runtime::GlobalCustomKernels().end()) {
      previous_ = existing->second;
    }
    core::runtime::RegisterGlobalCustomKernel(
        "", "Add", [](const NodeProto &node, core::runtime::RuntimeContext &runtime) {
          const core::runtime::Tensor &left = runtime.tensors().at(node.input()[0].value());
          const core::runtime::Tensor &right = runtime.tensors().at(node.input()[1].value());
          runtime.tensors()[node.output()[0].value()] = core::runtime::Tensor::FromFloat(
              node.output()[0].value(), left.shape, {left.AsFloat()[0] + right.AsFloat()[0]});
        });
  }

  ~ScopedAddKernel() {
    if (previous_) {
      core::runtime::RegisterGlobalCustomKernel("", "Add", std::move(previous_));
    } else {
      core::runtime::UnregisterGlobalCustomKernel("", "Add");
    }
  }

private:
  core::runtime::CustomKernelFn previous_;
};

} // namespace

TEST(PatternOptimization, IgnoresNullPositionalPlaceholders) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Identity", {"x"}, {"y"});
  builder.MakeOutput("y");

  auto pattern = std::make_shared<NullPlaceholderPattern>();
  core::builder::GraphGraph graph(
      builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{pattern});
  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize(1);

  ASSERT_EQ(rewrites.size(), 1u);
  EXPECT_EQ(rewrites[0].matched_nodes, std::vector<std::size_t>({0}));
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Relu");
}

TEST(PatternOptimization, FastOpTypeIsCachedAcrossOptimizationIterations) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Add", {"x", "x"}, {"y"});
  builder.MakeOutput("y");

  auto pattern = std::make_shared<CountingFastOpTypePattern>();
  std::vector<std::shared_ptr<core::builder::PatternOptimization>> patterns = {pattern};
  core::builder::GraphGraph graph(builder, std::move(patterns));

  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize(4);

  EXPECT_EQ(rewrites.size(), 4u);
  EXPECT_EQ(pattern->fast_op_type_calls, 1u);
}

TEST(PatternOptimization, CastCastCollapsesRedundantOuterCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  EXPECT_EQ(pattern.priority, 1);
  EXPECT_EQ(pattern.FastOpType(), std::set<std::string>({"Cast"}));
  EXPECT_EQ(pattern.ToString(),
            "PatternOptimization(name=CastCast, priority=1, fast_op_types=[Cast])");

  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, &pattern);
  ASSERT_EQ(match.nodes.size(), 2u);
  EXPECT_EQ(match.insert_at, match.nodes[1]);
  EXPECT_EQ(match.ToString(), "MatchResult(pattern=CastCast, nodes=[Cast(outputs=[middle]), "
                              "Cast(outputs=[y])], insert_at=Cast(outputs=[y]))");

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Cast");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  ASSERT_NE(FindAttribute(replacements[0], "to"), nullptr);
  EXPECT_EQ(FindAttribute(replacements[0], "to")->i(),
            static_cast<int64_t>(TensorProto::DataType::FLOAT));
}

TEST(PatternOptimization, CastReplacesRedundantConversionWithIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "y", TensorProto::DataType::FLOAT, "redundant");
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastPattern pattern;
  EXPECT_EQ(pattern.priority, 0);
  EXPECT_EQ(pattern.FastOpType(), std::set<std::string>({"Cast"}));

  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.nodes, std::vector<const NodeProto *>({&builder.Nodes()[0]}));
  EXPECT_EQ(match.insert_at, &builder.Nodes()[0]);

  const utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].output()[0].value(), "y");
  EXPECT_EQ(replacements[0].name().value(), "CastPattern--redundant");
  EXPECT_TRUE(replacements[0].attribute().empty());
}

TEST(PatternOptimization, CastRejectsTypeChangingConversion) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "y", TensorProto::DataType::DOUBLE);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  EXPECT_EQ(match.pattern, nullptr);
  EXPECT_TRUE(match.nodes.empty());
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_TRUE(match.no_match->source_file.ends_with("cast_pattern.cc"));
  EXPECT_GT(match.no_match->source_line, 0u);
  EXPECT_EQ(match.no_match->reason, "the input and target element types differ");
  EXPECT_NE(match.ToString().find("cast_pattern.cc:"), std::string::npos);
  EXPECT_NE(match.ToString().find("the input and target element types differ"), std::string::npos);

  core::builder::OptimizationReport report;
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastPattern>());
  core::builder::GraphGraph diagnostic_graph(builder, std::move(patterns));
  const std::vector<core::builder::LocalRewriting> rewrites =
      diagnostic_graph.Optimize(-1, &report);
  EXPECT_TRUE(rewrites.empty());
  ASSERT_EQ(report.patterns.size(), 1u);
  ASSERT_EQ(report.patterns[0].no_matches.size(), 1u);
  EXPECT_TRUE(report.patterns[0].no_matches[0].source_file.ends_with("cast_pattern.cc"));
  EXPECT_EQ(report.patterns[0].no_matches[0].source_line, match.no_match->source_line);
  EXPECT_EQ(report.patterns[0].no_matches[0].reason, "the input and target element types differ");
  EXPECT_EQ(report.patterns[0].no_matches[0].occurrences, 1u);
}

TEST(PatternOptimization, ApplyContractErrorIncludesSourceLocation) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastPattern pattern;

  try {
    pattern.Apply(graph, {});
    FAIL() << "Expected CastPattern::Apply to reject an invalid direct call.";
  } catch (const core::builder::BuilderError &error) {
    EXPECT_TRUE(error.SourceFile().ends_with("cast_pattern.cc"));
    EXPECT_GT(error.SourceLine(), 0u);
    EXPECT_NE(std::string(error.what()).find("cast_pattern.cc:"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("expects one default-domain Cast node"),
              std::string::npos);
  }
}

TEST(PatternOptimization, OptimizeCastProducesExpectedGraph) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  builder.MakeNode("Neg", {"middle"}, {"y"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "y");
}

TEST(PatternOptimization, AcceptsCustomPriority) {
  onnx_patterns::CastCastPattern pattern(3);
  EXPECT_EQ(pattern.priority, 3);
}

TEST(PatternOptimization, CastCastUsesIdentityForSafeRoundTrip) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT16);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_TRUE(replacements[0].attribute().empty());
}

TEST(PatternOptimization, CastCastKeepsSharedInnerCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeNode("Identity", {"middle"}, {"other"});
  builder.MakeOutput("y");
  builder.MakeOutput("other");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, nullptr);

  utils::RepeatedProtoField<NodeProto> replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].output()[0].value(), "middle");
  EXPECT_EQ(replacements[1].input()[0].value(), "x");
}

TEST(PatternOptimization, CastCastRejectsLossyRoundTrip) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::INT32);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::CastCastPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[1]);
  EXPECT_EQ(match.pattern, nullptr);
  EXPECT_TRUE(match.nodes.empty());
}

TEST(PatternOptimization, OptimizeAppliesPatternAndCleanupPasses) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeNode("Identity", {"x"}, {"unused"});
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));

  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize();
  ASSERT_EQ(rewrites.size(), 2u);
  ASSERT_NE(rewrites[0].pattern, nullptr);
  EXPECT_EQ(rewrites[0].pattern->Name(), "CastCast");
  EXPECT_EQ(rewrites[0].matched_nodes, std::vector<std::size_t>({0, 1}));
  EXPECT_EQ(rewrites[0].added_nodes.size(), 1u);
  EXPECT_TRUE(rewrites[0].added_initializers.empty());
  EXPECT_EQ(rewrites[0].insert_at, 1u);
  EXPECT_EQ(rewrites[0].iteration, 0u);
  EXPECT_GE(rewrites[0].match_time_ns, 0);
  EXPECT_GE(rewrites[0].apply_time_ns, 0);
  EXPECT_EQ(rewrites[0].ToString(),
            "LocalRewriting(pattern=CastCast, graph_path=[], matched_nodes=[0, 1], "
            "added_nodes=[Cast(outputs=[y])], added_initializers=[], "
            "added_initializer_positions=[], removed_initializers=[], value_renames=[], "
            "insert_at=1, iteration=0, match_time_ns=" +
                std::to_string(rewrites[0].match_time_ns) +
                ", apply_time_ns=" + std::to_string(rewrites[0].apply_time_ns) + ")");
  ASSERT_NE(rewrites[1].pattern, nullptr);
  EXPECT_EQ(rewrites[1].pattern->Name(), "RemoveIdentityNodes");
  EXPECT_EQ(rewrites[1].iteration, 1u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Cast");
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "y");
}

TEST(PatternOptimization, CleanupPassesProduceReplayableRewritingSequence) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeInitializer(MakeInitializer<float>("c1", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("c2", {1}, {1.0f}));
  builder.MakeNode("Neg", {"x"}, {"a"});
  builder.MakeNode("Neg", {"x"}, {"b"});
  builder.MakeNode("Identity", {"b"}, {"forwarded"});
  builder.MakeNode("Add", {"a", "forwarded"}, {"y"});
  builder.MakeNode("Relu", {"x"}, {"dead"});
  builder.MakeOutput("y");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();

  core::builder::GraphGraph graph(
      builder, std::vector<std::unique_ptr<core::builder::PatternOptimization>>{});
  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize();

  ASSERT_EQ(rewrites.size(), 4u);
  EXPECT_EQ(rewrites[0].pattern->Name(), "RemoveDuplicateNodes");
  EXPECT_EQ(rewrites[1].pattern->Name(), "RemoveIdentityNodes");
  EXPECT_EQ(rewrites[2].pattern->Name(), "RemoveUnusedNodes");
  EXPECT_EQ(rewrites[3].pattern->Name(), "RemoveDuplicateInitializers");
  ASSERT_EQ(rewrites[0].value_renames.size(), 1u);
  EXPECT_EQ(rewrites[0].value_renames[0].first, "b");
  EXPECT_EQ(rewrites[0].value_renames[0].second, "a");
  ASSERT_EQ(rewrites[1].value_renames.size(), 1u);
  EXPECT_EQ(rewrites[1].value_renames[0].first, "forwarded");
  EXPECT_EQ(rewrites[1].value_renames[0].second, "a");
  ASSERT_EQ(rewrites[3].value_renames.size(), 1u);
  EXPECT_EQ(rewrites[3].value_renames[0].first, "c2");
  EXPECT_EQ(rewrites[3].value_renames[0].second, "c1");
  for (std::size_t i = 0; i < rewrites.size(); ++i) {
    EXPECT_EQ(rewrites[i].iteration, i);
  }
  EXPECT_EQ(rewrites[3].removed_initializers, std::vector<std::size_t>({0, 1}));
  ASSERT_EQ(rewrites[3].added_initializers.size(), 1u);
  EXPECT_EQ(rewrites[3].added_initializers[0].name().value(), "c1");

  const GraphProto optimized = builder.BuildGraph();
  const GraphProto replayed = core::builder::Replay(original, rewrites, SchemaLookup());
  EXPECT_EQ(replayed.SerializeAsString(), optimized.SerializeAsString());
}

TEST(PatternOptimization, OptimizeReportsPhaseAndPatternTimings) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::OptimizationReport report;

  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize(-1, &report);
  ASSERT_EQ(rewrites.size(), 1u);
  EXPECT_EQ(report.rewrites, 1u);
  EXPECT_EQ(report.iterations, 2u);
  EXPECT_GE(report.matching_time_ns, rewrites[0].match_time_ns);
  EXPECT_GE(report.rewriting_time_ns, rewrites[0].apply_time_ns);
  EXPECT_GE(report.cleanup_time_ns, 0);
  EXPECT_GE(report.constant_folding_time_ns, 0);
  EXPECT_EQ(report.subgraph_optimization_time_ns, 0);
  ASSERT_EQ(report.patterns.size(), 1u);
  EXPECT_EQ(report.patterns[0].pattern_name, "CastCast");
  EXPECT_EQ(report.patterns[0].attempts, 3u);
  EXPECT_EQ(report.patterns[0].matches, 1u);
  EXPECT_GE(report.patterns[0].match_time_ns, rewrites[0].match_time_ns);
  EXPECT_EQ(report.patterns[0].apply_time_ns, rewrites[0].apply_time_ns);
  EXPECT_EQ(report.TotalTimeNs(), report.matching_time_ns + report.rewriting_time_ns +
                                      report.cleanup_time_ns + report.constant_folding_time_ns);
  EXPECT_NE(report.ToString().find("phases={matching:"), std::string::npos);
  EXPECT_NE(report.ToString().find("CastCast(attempts=3, matches=1"), std::string::npos);
}

TEST(PatternOptimization, OptimizesNestedSubgraphsAndReplaysByGraphPath) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  builder.MakeNode("Neg", {"x"}, {"captured"});

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  AddCast(body, "captured", "body_middle", TensorProto::DataType::FLOAT);
  AddCast(body, "body_middle", "body_y", TensorProto::DataType::FLOAT);
  body.MakeOutput("body_y");

  core::builder::GraphBuilder &leaf = body.MakeSubgraph("leaf");
  AddCast(leaf, "captured", "leaf_middle", TensorProto::DataType::FLOAT);
  AddCast(leaf, "leaf_middle", "leaf_y", TensorProto::DataType::FLOAT);
  leaf.MakeOutput("leaf_y");
  AddSubgraphReference(body, "body", "leaf", "leaf_result");
  body.MakeOutput("leaf_result");

  AddSubgraphReference(builder, "body", "body", "body_result");
  builder.MakeOutput("body_result");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  OperatorSetIdProto *opset = original.add_opset_import();
  opset->set_domain("");
  opset->set_version(21);

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::OptimizationReport report;
  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize(-1, &report);

  ASSERT_EQ(rewrites.size(), 2u);
  EXPECT_EQ(rewrites[0].graph_path, std::vector<std::string>({"body", "leaf"}));
  EXPECT_EQ(rewrites[1].graph_path, std::vector<std::string>({"body"}));
  EXPECT_EQ(rewrites[0].iteration, 0u);
  EXPECT_EQ(rewrites[1].iteration, 1u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 2u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].op_type().value(), "Cast");
  ASSERT_EQ(builder.Subgraph("body").Subgraph("leaf").Nodes().size(), 1u);
  EXPECT_EQ(builder.Subgraph("body").Subgraph("leaf").Nodes()[0].op_type().value(), "Cast");

  EXPECT_EQ(report.rewrites, 2u);
  EXPECT_GT(report.subgraph_optimization_time_ns, 0);
  ASSERT_EQ(report.patterns.size(), 1u);
  EXPECT_EQ(report.patterns[0].matches, 2u);
  ASSERT_EQ(report.subgraphs.size(), 2u);
  EXPECT_EQ(report.subgraphs[0].graph_path, std::vector<std::string>({"body"}));
  EXPECT_EQ(report.subgraphs[0].rewrites, 2u);
  EXPECT_EQ(report.subgraphs[1].graph_path, std::vector<std::string>({"body", "leaf"}));
  EXPECT_EQ(report.subgraphs[1].rewrites, 1u);
  EXPECT_NE(report.ToString().find("Subgraph(path=body/leaf"), std::string::npos);
  EXPECT_EQ(report.TotalTimeNs(), report.matching_time_ns + report.rewriting_time_ns +
                                      report.cleanup_time_ns + report.constant_folding_time_ns +
                                      report.subgraph_optimization_time_ns);

  const GraphProto optimized = builder.BuildGraph();
  const GraphProto replayed = core::builder::Replay(original, rewrites, SchemaLookup());
  EXPECT_EQ(replayed.SerializeAsString(), optimized.SerializeAsString());
}

TEST(PatternOptimization, SubgraphPatternsSeeCapturedParentInitializers) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion(kDefaultOnnxDomain, 21);
  builder.MakeInitializer(MakeInitializer<float>("a", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {1}, {2.0f}));

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeNode("Identity", {"a"}, {"body_y"});
  body.MakeOutput("body_y");
  AddSubgraphReference(builder, "body", "body", "body_result");
  builder.MakeOutput("body_result");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();
  OperatorSetIdProto *opset = original.add_opset_import();
  opset->set_domain("");
  opset->set_version(21);

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReplaceIdentityWithConstantAddPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize();

  ASSERT_EQ(rewrites.size(), 1u);
  EXPECT_EQ(rewrites[0].graph_path, std::vector<std::string>({"body"}));
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 1u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].op_type().value(), "Add");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].input()[0].value(), "a");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].input()[1].value(), "b");

  const GraphProto optimized = builder.BuildGraph();
  const GraphProto replayed = core::builder::Replay(original, rewrites, SchemaLookup());
  EXPECT_EQ(replayed.SerializeAsString(), optimized.SerializeAsString());
}

TEST(PatternOptimization, OptimizeFoldsConstantReplacementAndReplaysInitializer) {
  ScopedAddKernel kernel;
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInitializer(MakeInitializer<float>("a", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("b", {1}, {2.0f}));
  builder.MakeNode("Add", {"a", "b"}, {"untouched"});
  builder.MakeNode("Identity", {"a"}, {"y"});
  builder.MakeOutput("untouched");
  builder.MakeOutput("y");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<ReplaceIdentityWithConstantAddPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  core::builder::OptimizationReport report;

  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize(-1, &report);
  ASSERT_EQ(rewrites.size(), 1u);
  EXPECT_TRUE(rewrites[0].added_nodes.empty());
  ASSERT_EQ(rewrites[0].added_initializers.size(), 1u);
  EXPECT_EQ(rewrites[0].added_initializers[0].name().value(), "y");
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "untouched");
  EXPECT_TRUE(graph.IsConstant("y"));
  const TensorProto *folded = graph.GetComputedConstant("y");
  ASSERT_NE(folded, nullptr);
  EXPECT_FLOAT_EQ(core::runtime::TensorFromProto(*folded).AsFloat()[0], 3.0f);
  EXPECT_GT(report.constant_folding_time_ns, 0);

  const GraphProto replayed = core::builder::Replay(original, rewrites, SchemaLookup());
  ASSERT_EQ(replayed.node_size(), 1);
  EXPECT_EQ(replayed.node(0).output()[0].value(), "untouched");
  ASSERT_EQ(replayed.initializer_size(), 3);
  EXPECT_EQ(replayed.initializer(2).name().value(), "y");
  EXPECT_FLOAT_EQ(core::runtime::TensorFromProto(replayed.initializer(2)).AsFloat()[0], 3.0f);
}

TEST(PatternOptimization, OptimizeAppliesDisjointMatchesInOneIteration) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x1", core::symbolic::TensorType::kFloat16, Shape());
  builder.MakeInput("x2", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x1", "middle1", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle1", "y1", TensorProto::DataType::FLOAT);
  AddCast(builder, "x2", "middle2", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle2", "y2", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y1");
  builder.MakeOutput("y2");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));

  const std::vector<core::builder::LocalRewriting> rewrites = graph.Optimize(1);
  ASSERT_EQ(rewrites.size(), 2u);
  EXPECT_EQ(rewrites[0].iteration, 0u);
  EXPECT_EQ(rewrites[1].iteration, 0u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].input()[0].value(), "x1");
  EXPECT_EQ(builder.Nodes()[1].input()[0].value(), "x2");

  const GraphProto replayed = core::builder::Replay(original, rewrites, SchemaLookup());
  ASSERT_EQ(replayed.node_size(), 2);
  EXPECT_EQ(replayed.node(0).input()[0].value(), "x1");
  EXPECT_EQ(replayed.node(0).output()[0].value(), "y1");
  EXPECT_EQ(replayed.node(1).input()[0].value(), "x2");
  EXPECT_EQ(replayed.node(1).output()[0].value(), "y2");
}

TEST(PatternOptimization, ReplayRestoresAddedInitializers) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape());
  builder.MakeNode("Identity", {"x"}, {"y"});
  builder.MakeOutput("y");

  ModelProto original;
  *original.mutable_graph() = builder.BuildGraph();

  core::builder::LocalRewriting rewrite;
  rewrite.pattern = std::make_shared<onnx_patterns::CastCastPattern>();
  rewrite.matched_nodes = {0};
  TensorProto &weight = rewrite.added_initializers.add();
  weight.set_name("weight");
  weight.set_data_type(TensorProto::DataType::FLOAT);
  weight.add_dims(2);
  weight.add_float_data(1.0f);
  weight.add_float_data(2.0f);
  rewrite.added_nodes.add() = MakeNode("Add", {"x", "weight"}, {"y"});

  const GraphProto replayed = core::builder::Replay(
      original, std::vector<core::builder::LocalRewriting>{rewrite}, SchemaLookup());
  ASSERT_EQ(replayed.initializer_size(), 1);
  EXPECT_EQ(replayed.initializer(0).name().value(), "weight");
  ASSERT_EQ(replayed.node_size(), 1);
  EXPECT_EQ(replayed.node(0).op_type().value(), "Add");
  EXPECT_EQ(replayed.node(0).input()[1].value(), "weight");
}

TEST(PatternOptimization, LocalRewritingKeepsPatternAlive) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  std::vector<core::builder::LocalRewriting> rewrites;
  {
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
    core::builder::GraphGraph graph(builder, std::move(patterns));
    rewrites = graph.Optimize();
  }

  ASSERT_EQ(rewrites.size(), 1u);
  ASSERT_NE(rewrites[0].pattern, nullptr);
  EXPECT_EQ(rewrites[0].pattern->Name(), "CastCast");
}

TEST(PatternOptimization, OptimizeKeepsSharedInnerCastInTopologicalOrder) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  builder.MakeNode("Neg", {"middle"}, {"other"});
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");
  builder.MakeOutput("other");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));

  EXPECT_EQ(graph.Optimize().size(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "middle");
  EXPECT_EQ(builder.Nodes()[1].output()[0].value(), "y");
  EXPECT_EQ(builder.Nodes()[2].op_type().value(), "Neg");
}

TEST(PatternOptimization, OptimizeHonorsDoNotRemovePredicate) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape());
  AddCast(builder, "x", "middle", TensorProto::DataType::FLOAT);
  AddCast(builder, "middle", "y", TensorProto::DataType::FLOAT);
  builder.MakeOutput("y");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::CastCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns), [](const NodeProto &node) {
    return node.output()[0].value() == "middle";
  });

  EXPECT_TRUE(graph.Optimize().empty());
  EXPECT_EQ(builder.Nodes().size(), 2u);
}

TEST(PatternOptimization, RegistersBuiltInPatternsOnce) {
  onnx_patterns::RegisterPatterns();
  const std::vector<std::string> names = core::builder::RegisteredPatternNames();
  EXPECT_EQ(std::count(names.begin(), names.end(), "Cast"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "CastCast"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "CastCastBinary"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "CastOpCast"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ClipClip"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ConstantToInitializer"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "ConvBiasNull"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "Dropout"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "Identity"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "NotNot"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "PadConv"), 1);

  onnx_patterns::RegisterPatterns();
  EXPECT_EQ(core::builder::RegisteredPatternNames(), names);

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns =
      core::builder::CreateRegisteredPatterns();
  const bool found_cast = std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
    return dynamic_cast<onnx_patterns::CastPattern *>(pattern.get()) != nullptr;
  });
  const bool found_cast_cast =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::CastCastPattern *>(pattern.get()) != nullptr;
      });
  const bool found_cast_cast_binary =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::CastCastBinaryPattern *>(pattern.get()) != nullptr;
      });
  const bool found_cast_op_cast =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::CastOpCastPattern *>(pattern.get()) != nullptr;
      });
  const bool found_clip_clip =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::ClipClipPattern *>(pattern.get()) != nullptr;
      });
  const bool found_constant_to_initializer =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::ConstantToInitializerPattern *>(pattern.get()) !=
               nullptr;
      });
  const bool found_conv_bias_null =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::ConvBiasNullPattern *>(pattern.get()) != nullptr;
      });
  const bool found_dropout = std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
    return dynamic_cast<onnx_patterns::DropoutPattern *>(pattern.get()) != nullptr;
  });
  const bool found_identity =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::IdentityPattern *>(pattern.get()) != nullptr;
      });
  const bool found_not_not = std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
    return dynamic_cast<onnx_patterns::NotNotPattern *>(pattern.get()) != nullptr;
  });
  const bool found_pad_conv =
      std::any_of(patterns.begin(), patterns.end(), [](const auto &pattern) {
        return dynamic_cast<onnx_patterns::PadConvPattern *>(pattern.get()) != nullptr;
      });
  EXPECT_TRUE(found_cast);
  EXPECT_TRUE(found_cast_cast);
  EXPECT_TRUE(found_cast_cast_binary);
  EXPECT_TRUE(found_cast_op_cast);
  EXPECT_TRUE(found_clip_clip);
  EXPECT_TRUE(found_constant_to_initializer);
  EXPECT_TRUE(found_conv_bias_null);
  EXPECT_TRUE(found_dropout);
  EXPECT_TRUE(found_identity);
  EXPECT_TRUE(found_not_not);
  EXPECT_TRUE(found_pad_conv);
}

} // namespace Test
