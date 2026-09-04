// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/matmul/matmul_pattern.h"
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

core::symbolic::SymShape SymbolicShape(const std::vector<core::symbolic::SymDim> &dims) {
  return core::symbolic::SymShape(dims);
}

void AddShape(core::builder::GraphBuilder &builder, const std::string &name,
              const std::vector<int64_t> &values) {
  builder.MakeInitializer(MakeInitializerShape(name.c_str(), values));
}

void AddFloat(core::builder::GraphBuilder &builder, const std::string &name,
              const std::vector<int64_t> &dims, const std::vector<float> &values) {
  builder.MakeInitializer(MakeInitializer<float>(name.c_str(), dims, values));
}

utils::RepeatedProtoField<AttributeProto> PermAttr(const std::vector<int64_t> &perm) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name("perm");
  attribute.set_type(AttributeProto::AttributeType::INTS);
  for (int64_t value : perm) {
    attribute.ints().push_back(value);
  }
  return attributes;
}

utils::RepeatedProtoField<AttributeProto> IntAttr(const char *name, int64_t value) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(value);
  return attributes;
}

utils::RepeatedProtoField<AttributeProto> FloatAttr(const char *name, float value) {
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &attribute = attributes.add();
  attribute.set_name(name);
  attribute.set_type(AttributeProto::AttributeType::FLOAT);
  attribute.set_f(value);
  return attributes;
}

std::vector<int64_t> AttributeInts(const NodeProto &node, const char *name) {
  std::vector<int64_t> values;
  GetAttributeInts(node, name, values);
  return values;
}

const TensorProto *FindInitializer(const core::builder::GraphBuilder &builder,
                                   const std::string &name) {
  for (const TensorProto &initializer : builder.Initializers()) {
    if (initializer.name().value() == name) {
      return &initializer;
    }
  }
  return nullptr;
}

void SeedShape(core::builder::GraphBuilder &builder, const std::string &name,
               const std::vector<int64_t> &dims) {
  auto &shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
  shapes.Set(name,
             core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, Shape(dims)));
}

void SetNodeDomain(core::builder::GraphBuilder &builder, std::size_t index,
                   const std::string &domain) {
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  nodes[index].set_domain(domain);
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

TEST(MatMulAddPattern, FusesMatMulAddAndRejectsSharedMatMulOutput) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  builder.MakeInput("bias", core::symbolic::TensorType::kFloat, Shape({4}));
  builder.MakeNode("MatMul", {"x", "w"}, {"mm"}, "", "mm");
  builder.MakeNode("Add", {"mm", "bias"}, {"out"}, "", "add");
  core::builder::GraphGraph graph(builder);
  onnx_patterns::MatMulAddPattern pattern;
  const core::builder::MatchResult match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[1]);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gemm");
  EXPECT_EQ(replacements[0].name().value(), "MatMulAddPattern--mm");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "w");
  EXPECT_EQ(replacements[0].input()[2].value(), "bias");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  rejected.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  rejected.MakeInput("bias", core::symbolic::TensorType::kFloat, Shape({4}));
  rejected.MakeNode("MatMul", {"x", "w"}, {"mm"});
  rejected.MakeNode("Add", {"mm", "bias"}, {"out"});
  rejected.MakeNode("Identity", {"mm"}, {"other"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[0]).pattern, nullptr);
}

TEST(GemmSumFusionPattern, UsesStandardBiasAndRejectsExistingBias) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  builder.MakeInput("sum", core::symbolic::TensorType::kFloat, Shape({4}));
  builder.MakeNode("Gemm", {"x", "w"}, {"gemm"}, "", "gemm", FloatAttr("beta", 0.5F));
  builder.MakeNode("Sum", {"sum", "gemm"}, {"out"});
  SeedShape(builder, "gemm", {2, 4});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::GemmSumFusionPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gemm");
  EXPECT_EQ(replacements[0].input()[2].value(), "sum");
  EXPECT_EQ(GetAttributeOr<float>(replacements[0], "beta", 1.0F), 1.0F);

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  rejected.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  rejected.MakeInput("bias", core::symbolic::TensorType::kFloat, Shape({4}));
  rejected.MakeInput("sum", core::symbolic::TensorType::kFloat, Shape({4}));
  rejected.MakeNode("Gemm", {"x", "w", "bias"}, {"gemm"});
  rejected.MakeNode("Sum", {"gemm", "sum"}, {"out"});
  SeedShape(rejected, "gemm", {2, 4});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

TEST(GemmTransposePattern, InsertsWeightTransposeAndRejectsExistingTranspose) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloat(builder, "w", {3, 4}, std::vector<float>(12, 1.0F));
  builder.MakeNode("Gemm", {"x", "w"}, {"out"}, "", "gemm", FloatAttr("alpha", 2.0F));
  core::builder::GraphGraph graph(builder);
  onnx_patterns::GemmTransposePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[0]);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[0].input()[0].value(), "w");
  EXPECT_EQ(AttributeInts(replacements[0], "perm"), (std::vector<int64_t>{1, 0}));
  EXPECT_EQ(replacements[1].op_type().value(), "Gemm");
  EXPECT_EQ(replacements[1].input()[1].value(), replacements[0].output()[0].value());
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[1], "transB", 0), 1);
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(replacements[1], "alpha", 1.0F), 2.0F);

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloat(rejected, "w", {3, 4}, std::vector<float>(12, 1.0F));
  rejected.MakeNode("Gemm", {"x", "w"}, {"out"}, "", "", IntAttr("transB", 1));
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[0]).pattern, nullptr);
}

TEST(MatMulReshape2Of3Pattern, MovesBatchReshapeAndRejectsOnlyOneReshape) {
  core::builder::GraphBuilder seed("positive", SchemaLookup());
  seed.MakeInput("left", core::symbolic::TensorType::kFloat, Shape({13, 4, 7, 7}));
  seed.MakeInput("right", core::symbolic::TensorType::kFloat, Shape({52, 7, 8}));
  AddShape(seed, "left_shape", {52, 7, 7});
  AddShape(seed, "out_shape", {13, 4, 7, 8});
  seed.MakeNode("Reshape", {"left", "left_shape"}, {"left3"});
  seed.MakeNode("MatMul", {"left3", "right"}, {"mm"}, "", "mm");
  seed.MakeNode("Reshape", {"mm", "out_shape"}, {"out"});
  seed.MakeOutput("out", core::symbolic::TensorType::kFloat, Shape({13, 4, 7, 8}));
  core::builder::GraphBuilder builder = std::move(seed);
  SeedShape(builder, "left3", {52, 7, 7});
  SeedShape(builder, "mm", {52, 7, 8});
  SeedShape(builder, "out", {13, 4, 7, 8});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::MatMulReshape2Of3Pattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  EXPECT_EQ(match.insert_at, nullptr);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[0].input()[0].value(), "right");
  EXPECT_EQ(replacements[1].op_type().value(), "MatMul");
  EXPECT_EQ(replacements[1].input()[0].value(), "left");
  EXPECT_EQ(replacements[1].input()[1].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[1].output()[0].value(), "out");

  core::builder::GraphBuilder rejected_seed("rejected", SchemaLookup());
  rejected_seed.MakeInput("left", core::symbolic::TensorType::kFloat, Shape({13, 4, 7, 7}));
  rejected_seed.MakeInput("right", core::symbolic::TensorType::kFloat, Shape({52, 7, 8}));
  AddShape(rejected_seed, "left_shape", {52, 7, 7});
  rejected_seed.MakeNode("Reshape", {"left", "left_shape"}, {"left3"});
  rejected_seed.MakeNode("MatMul", {"left3", "right"}, {"out"});
  rejected_seed.MakeOutput("out", core::symbolic::TensorType::kFloat, Shape({52, 7, 8}));
  core::builder::GraphBuilder rejected = std::move(rejected_seed);
  SeedShape(rejected, "left3", {52, 7, 7});
  SeedShape(rejected, "out", {52, 7, 8});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

TEST(MulMulMatMulPattern, MovesScalarProductAfterMatMulAndRejectsBroadcastScalar) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({32, 16}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({16, 64}));
  AddFloat(builder, "c1", {}, {0.4F});
  AddFloat(builder, "c2", {1}, {0.6F});
  builder.MakeNode("Mul", {"x", "c1"}, {"left"});
  builder.MakeNode("Mul", {"c2", "y"}, {"right"});
  builder.MakeNode("MatMul", {"left", "right"}, {"out"}, "", "mm");
  core::builder::GraphGraph graph(builder);
  onnx_patterns::MulMulMatMulPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, nullptr);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "MatMul");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "y");
  EXPECT_EQ(replacements[1].op_type().value(), "Mul");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
  const TensorProto *product = FindInitializer(builder, replacements[1].input()[1].value());
  ASSERT_NE(product, nullptr);
  std::vector<double> product_values;
  ASSERT_TRUE(ReadFloatingValues(*product, product_values));
  ASSERT_EQ(product_values.size(), 1u);
  EXPECT_FLOAT_EQ(static_cast<float>(product_values[0]), 0.24F);

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({32, 16}));
  rejected.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({16, 64}));
  AddFloat(rejected, "c1", {}, {0.4F});
  AddFloat(rejected, "c2", {1, 1}, {0.6F});
  rejected.MakeNode("Mul", {"x", "c1"}, {"left"});
  rejected.MakeNode("Mul", {"c2", "y"}, {"right"});
  rejected.MakeNode("MatMul", {"left", "right"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[2]).pattern, nullptr);
}

TEST(MatMulBatchNormalizationFusionPattern, FoldsConstantsAndRejectsDynamicWeight) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloat(builder, "w", {3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  AddFloat(builder, "scale", {2}, {2.0F, 3.0F});
  AddFloat(builder, "bias", {2}, {0.5F, -0.5F});
  AddFloat(builder, "mean", {2}, {1.0F, 2.0F});
  AddFloat(builder, "variance", {2}, {4.0F, 9.0F});
  builder.MakeNode("MatMul", {"x", "w"}, {"mm"}, "", "mm");
  builder.MakeNode("BatchNormalization", {"mm", "scale", "bias", "mean", "variance"},
                   {"out", "", ""});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::MatMulBatchNormalizationFusionPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gemm");
  ASSERT_EQ(replacements[0].input_size(), 3);
  EXPECT_NE(FindInitializer(builder, replacements[0].input()[1].value()), nullptr);
  EXPECT_NE(FindInitializer(builder, replacements[0].input()[2].value()), nullptr);

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  rejected.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 2}));
  AddFloat(rejected, "scale", {2}, {1.0F, 1.0F});
  AddFloat(rejected, "bias", {2}, {0.0F, 0.0F});
  AddFloat(rejected, "mean", {2}, {0.0F, 0.0F});
  AddFloat(rejected, "variance", {2}, {1.0F, 1.0F});
  rejected.MakeNode("MatMul", {"x", "w"}, {"mm"});
  rejected.MakeNode("BatchNormalization", {"mm", "scale", "bias", "mean", "variance"},
                    {"out", "", ""});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

TEST(MatMulBatchNormalizationFusionPattern, RejectsTrainingOutputsBeforeOpset14) {
  core::builder::GraphBuilder builder("training", SchemaLookup());
  builder.SetOpsetVersion("", 13);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloat(builder, "w", {3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  AddFloat(builder, "scale", {2}, {1.0F, 1.0F});
  AddFloat(builder, "bias", {2}, {0.0F, 0.0F});
  AddFloat(builder, "mean", {2}, {0.0F, 0.0F});
  AddFloat(builder, "variance", {2}, {1.0F, 1.0F});
  builder.MakeNode("MatMul", {"x", "w"}, {"mm"});
  builder.MakeNode("BatchNormalization", {"mm", "scale", "bias", "mean", "variance"},
                   {"out", "running_mean", "running_var", "saved_mean", "saved_var"});

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MatMulBatchNormalizationFusionPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(MatMulScaleFusionPattern, FoldsWeightAndRejectsUnsafeDivision) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloat(builder, "w", {3, 2}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  AddFloat(builder, "scale", {}, {0.5F});
  builder.MakeNode("MatMul", {"x", "w"}, {"mm"}, "", "mm");
  builder.MakeNode("Mul", {"mm", "scale"}, {"out"});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::MatMulScaleFusionPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "MatMul");
  EXPECT_NE(replacements[0].input()[1].value(), "w");
  EXPECT_NE(FindInitializer(builder, replacements[0].input()[1].value()), nullptr);

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  rejected.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 2}));
  AddFloat(rejected, "scale", {}, {2.0F});
  rejected.MakeNode("MatMul", {"x", "w"}, {"mm"});
  rejected.MakeNode("Div", {"scale", "mm"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[0]).pattern, nullptr);
}

TEST(ReshapeMatMulReshapePattern, RemovesThreeReshapesAndRejectsSharedMatMulOutput) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("left", core::symbolic::TensorType::kFloat, Shape({1, 1, 32, 128}));
  builder.MakeInput("right", core::symbolic::TensorType::kFloat, Shape({3, 5, 128, 64}));
  AddShape(builder, "left_shape", {1, 32, 128});
  AddShape(builder, "right_shape", {15, 128, 64});
  AddShape(builder, "out_shape", {3, 5, 32, 64});
  builder.MakeNode("Reshape", {"left", "left_shape"}, {"left3"});
  builder.MakeNode("Reshape", {"right", "right_shape"}, {"right3"});
  builder.MakeNode("MatMul", {"left3", "right3"}, {"mm"}, "", "mm");
  builder.MakeNode("Reshape", {"mm", "out_shape"}, {"out"});
  SeedShape(builder, "left3", {1, 32, 128});
  SeedShape(builder, "right3", {15, 128, 64});
  SeedShape(builder, "mm", {15, 32, 64});
  SeedShape(builder, "out", {3, 5, 32, 64});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReshapeMatMulReshapePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "MatMul");
  EXPECT_EQ(replacements[0].input()[0].value(), "left");
  EXPECT_EQ(replacements[0].input()[1].value(), "right");
  EXPECT_EQ(replacements[0].output()[0].value(), "out");
  EXPECT_EQ(replacements[0].name().value(), "ReshapeMatMulReshapePattern--mm");

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("left", core::symbolic::TensorType::kFloat, Shape({1, 1, 32, 128}));
  rejected.MakeInput("right", core::symbolic::TensorType::kFloat, Shape({3, 5, 128, 64}));
  AddShape(rejected, "left_shape", {1, 32, 128});
  AddShape(rejected, "right_shape", {15, 128, 64});
  AddShape(rejected, "out_shape", {3, 5, 32, 64});
  rejected.MakeNode("Reshape", {"left", "left_shape"}, {"left3"});
  rejected.MakeNode("Reshape", {"right", "right_shape"}, {"right3"});
  rejected.MakeNode("MatMul", {"left3", "right3"}, {"mm"});
  rejected.MakeNode("Reshape", {"mm", "out_shape"}, {"out"});
  rejected.MakeNode("Identity", {"mm"}, {"other"});
  SeedShape(rejected, "left3", {1, 32, 128});
  SeedShape(rejected, "right3", {15, 128, 64});
  SeedShape(rejected, "mm", {15, 32, 64});
  SeedShape(rejected, "out", {3, 5, 32, 64});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[2]).pattern, nullptr);
}

TEST(TransposeMatMulPattern, ConvertsSharedTransposeMatMulAndRejectsUnexpectedPerm) {
  core::builder::GraphBuilder seed("positive", SchemaLookup());
  seed.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({128, 32}));
  seed.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({128, 64}));
  seed.MakeNode("Transpose", {"x"}, {"xt"}, "", "transpose", PermAttr({1, 0}));
  seed.MakeNode("MatMul", {"xt", "y"}, {"out"}, "", "mm");
  seed.MakeNode("Identity", {"xt"}, {"other"});
  seed.MakeOutput("out", core::symbolic::TensorType::kFloat, Shape({32, 64}));
  seed.MakeOutput("other", core::symbolic::TensorType::kFloat, Shape({32, 128}));
  core::builder::GraphBuilder builder = std::move(seed);
  SeedShape(builder, "xt", {32, 128});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeMatMulPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Gemm");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[0].input()[1].value(), "y");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "transA", 0), 1);
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "transB", 0), 0);
  EXPECT_EQ(replacements[1].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[1].output()[0].value(), "xt");

  core::builder::GraphBuilder rejected_seed("rejected", SchemaLookup());
  rejected_seed.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({32, 128}));
  rejected_seed.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({128, 64}));
  rejected_seed.MakeNode("Transpose", {"x"}, {"xt"}, "", "", PermAttr({0, 1}));
  rejected_seed.MakeNode("MatMul", {"xt", "y"}, {"out"});
  rejected_seed.MakeOutput("out", core::symbolic::TensorType::kFloat, Shape({32, 64}));
  core::builder::GraphBuilder rejected = std::move(rejected_seed);
  SeedShape(rejected, "xt", {32, 128});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[1]).pattern, nullptr);
}

TEST(TransposeReshapeMatMulPattern, MovesRightTransposeAfterReshapeAndRejectsSharedInput) {
  core::builder::GraphBuilder seed("positive", SchemaLookup());
  seed.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 2, 5, 7}));
  seed.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({4, 3, 7}));
  AddShape(seed, "shape", {2, 2, 7, 3});
  seed.MakeNode("Transpose", {"y"}, {"yt"}, "", "transpose", PermAttr({0, 2, 1}));
  seed.MakeNode("Reshape", {"yt", "shape"}, {"right"});
  seed.MakeNode("MatMul", {"x", "right"}, {"out"}, "", "mm");
  seed.MakeOutput("out", core::symbolic::TensorType::kFloat, Shape({2, 2, 5, 3}));
  core::builder::GraphBuilder builder = std::move(seed);
  SeedShape(builder, "yt", {4, 7, 3});
  SeedShape(builder, "right", {2, 2, 7, 3});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeReshapeMatMulPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern) << match.ToString();
  EXPECT_EQ(match.insert_at, &builder.Nodes()[2]);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[0].input()[0].value(), "y");
  const TensorProto *shape = FindInitializer(builder, replacements[0].input()[1].value());
  ASSERT_NE(shape, nullptr);
  std::vector<int64_t> shape_values;
  ASSERT_TRUE(ReadIntegerValues(*shape, shape_values));
  EXPECT_EQ(shape_values, (std::vector<int64_t>{2, 2, 3, 7}));
  EXPECT_EQ(replacements[1].op_type().value(), "Transpose");
  EXPECT_EQ(AttributeInts(replacements[1], "perm"), (std::vector<int64_t>{0, 1, 3, 2}));
  EXPECT_EQ(replacements[1].output()[0].value(), "right");
  EXPECT_EQ(replacements[2].op_type().value(), "MatMul");
  EXPECT_EQ(replacements[2].input()[1].value(), "right");

  core::builder::GraphBuilder rejected_seed("rejected", SchemaLookup());
  rejected_seed.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 2, 5, 7}));
  rejected_seed.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({4, 3, 7}));
  AddShape(rejected_seed, "shape", {2, 2, 7, 3});
  rejected_seed.MakeNode("Transpose", {"y"}, {"yt"}, "", "", PermAttr({0, 2, 1}));
  rejected_seed.MakeNode("Reshape", {"yt", "shape"}, {"right"});
  rejected_seed.MakeNode("MatMul", {"x", "right"}, {"out"});
  rejected_seed.MakeNode("Identity", {"right"}, {"other"});
  rejected_seed.MakeOutput("out", core::symbolic::TensorType::kFloat, Shape({2, 2, 5, 3}));
  rejected_seed.MakeOutput("other", core::symbolic::TensorType::kFloat, Shape({2, 2, 7, 3}));
  core::builder::GraphBuilder rejected = std::move(rejected_seed);
  SeedShape(rejected, "yt", {4, 7, 3});
  SeedShape(rejected, "right", {2, 2, 7, 3});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[2]).pattern, nullptr);
}

TEST(SwitchReshapeActivationPattern, MovesReluBeforeTransposeAndRejectsSharedMatMulOutput) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 2, 6, 5}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({3, 2, 5, 6}));
  builder.MakeNode("MatMul", {"x", "y"}, {"mm"}, "", "mm");
  builder.MakeNode("Transpose", {"mm"}, {"transposed"}, "", "transpose", PermAttr({0, 2, 1, 3}));
  builder.MakeNode("Relu", {"transposed"}, {"out"}, "", "relu");
  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwitchReshapeActivationPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[0]);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "MatMul");
  EXPECT_EQ(replacements[0].name().value(), "SwitchReshapeActivationPattern--mm");
  EXPECT_EQ(replacements[1].op_type().value(), "Relu");
  EXPECT_EQ(replacements[1].input()[0].value(), replacements[0].output()[0].value());
  EXPECT_EQ(replacements[2].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[2].input()[0].value(), replacements[1].output()[0].value());
  EXPECT_EQ(replacements[2].output()[0].value(), "out");
  EXPECT_EQ(AttributeInts(replacements[2], "perm"), (std::vector<int64_t>{0, 2, 1, 3}));

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 2, 6, 5}));
  rejected.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({3, 2, 5, 6}));
  rejected.MakeNode("MatMul", {"x", "y"}, {"mm"});
  rejected.MakeNode("Transpose", {"mm"}, {"transposed"}, "", "", PermAttr({0, 2, 1, 3}));
  rejected.MakeNode("Relu", {"transposed"}, {"out"});
  rejected.MakeNode("Identity", {"mm"}, {"other"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[2]).pattern, nullptr);
}

TEST(ShapeBasedMatMulToMulPattern, ReplacesSymbolicOuterProductAndRejectsNonUnitReduction) {
  core::builder::GraphBuilder builder("positive", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat,
                    SymbolicShape({core::symbolic::SymDim("a"), core::symbolic::SymDim("b"),
                                   core::symbolic::SymDim(1)}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat,
                    SymbolicShape({core::symbolic::SymDim("a"), core::symbolic::SymDim(1),
                                   core::symbolic::SymDim("c")}));
  builder.MakeNode("MatMul", {"x", "y"}, {"mm"}, "", "mm");
  builder.MakeNode("Transpose", {"mm"}, {"out"}, "", "transpose", PermAttr({0, 2, 1}));
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedMatMulToMulPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  EXPECT_EQ(match.insert_at, &builder.Nodes()[0]);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[1].op_type().value(), "Reshape");
  EXPECT_EQ(replacements[2].op_type().value(), "Mul");
  EXPECT_EQ(replacements[2].output()[0].value(), "out");
  const TensorProto *left_shape = FindInitializer(builder, replacements[0].input()[1].value());
  const TensorProto *right_shape = FindInitializer(builder, replacements[1].input()[1].value());
  ASSERT_NE(left_shape, nullptr);
  ASSERT_NE(right_shape, nullptr);
  std::vector<int64_t> left_values;
  std::vector<int64_t> right_values;
  ASSERT_TRUE(ReadIntegerValues(*left_shape, left_values));
  ASSERT_TRUE(ReadIntegerValues(*right_shape, right_values));
  EXPECT_EQ(left_values, (std::vector<int64_t>{0, 1, -1}));
  EXPECT_EQ(right_values, (std::vector<int64_t>{0, -1, 1}));

  core::builder::GraphBuilder rejected("rejected", SchemaLookup());
  rejected.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 2}));
  rejected.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 2, 4}));
  rejected.MakeNode("MatMul", {"x", "y"}, {"out"});
  core::builder::GraphGraph rejected_graph(rejected);
  EXPECT_EQ(pattern.Match(rejected_graph, rejected.Nodes()[0]).pattern, nullptr);
}

TEST(MatMulAddPattern, RejectsNonUnitGemmBetaAndSubgraphCapture) {
  onnx_patterns::MatMulAddPattern pattern;

  core::builder::GraphBuilder beta("beta", SchemaLookup());
  beta.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  beta.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  beta.MakeInput("bias", core::symbolic::TensorType::kFloat, Shape({4}));
  beta.MakeNode("Gemm", {"x", "w"}, {"mm"}, "", "", FloatAttr("beta", 2.0F));
  beta.MakeNode("Add", {"mm", "bias"}, {"out"});
  core::builder::GraphGraph beta_graph(beta);
  EXPECT_EQ(pattern.Match(beta_graph, beta.Nodes()[0]).pattern, nullptr);

  core::builder::GraphBuilder captured("captured", SchemaLookup());
  captured.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  captured.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  captured.MakeInput("bias", core::symbolic::TensorType::kFloat, Shape({4}));
  captured.MakeNode("MatMul", {"x", "w"}, {"mm"});
  captured.MakeNode("Add", {"mm", "bias"}, {"out"});
  core::builder::GraphBuilder &body = captured.MakeSubgraph("body");
  body.MakeNode("Identity", {"mm"}, {"body_out"});
  body.MakeOutput("body_out");
  AddSubgraphReference(captured, "body", "body", "body_result");
  core::builder::GraphGraph captured_graph(captured);
  EXPECT_TRUE(captured_graph.IsUsedBySubgraph("mm"));
  EXPECT_EQ(pattern.Match(captured_graph, captured.Nodes()[0]).pattern, nullptr);
}

TEST(MatMulReshape2Of3Pattern, RejectsMatrixResultAndBatchShapeChanges) {
  onnx_patterns::MatMulReshape2Of3Pattern pattern;

  core::builder::GraphBuilder matrix("matrix", SchemaLookup());
  matrix.MakeInput("left", core::symbolic::TensorType::kFloat, Shape({13, 4, 1, 49}));
  matrix.MakeInput("right", core::symbolic::TensorType::kFloat, Shape({52, 7, 8}));
  AddShape(matrix, "left_shape", {52, 7, 7});
  AddShape(matrix, "out_shape", {13, 4, 7, 8});
  matrix.MakeNode("Reshape", {"left", "left_shape"}, {"left3"});
  matrix.MakeNode("MatMul", {"left3", "right"}, {"mm"});
  matrix.MakeNode("Reshape", {"mm", "out_shape"}, {"out"});
  SeedShape(matrix, "left3", {52, 7, 7});
  SeedShape(matrix, "mm", {52, 7, 8});
  SeedShape(matrix, "out", {13, 4, 7, 8});
  core::builder::GraphGraph matrix_graph(matrix);
  EXPECT_EQ(pattern.Match(matrix_graph, matrix.Nodes()[1]).pattern, nullptr);

  core::builder::GraphBuilder result("result", SchemaLookup());
  result.MakeInput("left", core::symbolic::TensorType::kFloat, Shape({13, 4, 7, 7}));
  result.MakeInput("right", core::symbolic::TensorType::kFloat, Shape({52, 7, 8}));
  AddShape(result, "left_shape", {52, 7, 7});
  AddShape(result, "out_shape", {13, 4, 7, 8});
  result.MakeNode("Reshape", {"left", "left_shape"}, {"left3"});
  result.MakeNode("MatMul", {"left3", "right"}, {"mm"});
  result.MakeNode("Reshape", {"mm", "out_shape"}, {"out"});
  SeedShape(result, "left3", {52, 7, 7});
  SeedShape(result, "mm", {52, 7, 9});
  SeedShape(result, "out", {13, 4, 7, 8});
  core::builder::GraphGraph result_graph(result);
  EXPECT_EQ(pattern.Match(result_graph, result.Nodes()[1]).pattern, nullptr);

  core::builder::GraphBuilder batch("batch", SchemaLookup());
  batch.MakeInput("left", core::symbolic::TensorType::kFloat, Shape({1, 1, 7, 7}));
  batch.MakeInput("right", core::symbolic::TensorType::kFloat, Shape({3, 4, 7, 8}));
  AddShape(batch, "left_shape", {1, 7, 7});
  AddShape(batch, "right_shape", {12, 7, 8});
  batch.MakeNode("Reshape", {"left", "left_shape"}, {"left3"});
  batch.MakeNode("Reshape", {"right", "right_shape"}, {"right3"});
  batch.MakeNode("MatMul", {"left3", "right3"}, {"out"});
  batch.MakeOutput("out", core::symbolic::TensorType::kFloat, Shape({12, 7, 8}));
  SeedShape(batch, "left3", {1, 7, 7});
  SeedShape(batch, "right3", {12, 7, 8});
  SeedShape(batch, "out", {12, 7, 8});
  core::builder::GraphGraph batch_graph(batch);
  EXPECT_EQ(pattern.Match(batch_graph, batch.Nodes()[2]).pattern, nullptr);
}

TEST(ReshapeMatMulReshapePattern, RejectsIncompatibleOriginalBatchBroadcast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("left", core::symbolic::TensorType::kFloat, Shape({2, 6, 32, 128}));
  builder.MakeInput("right", core::symbolic::TensorType::kFloat, Shape({3, 4, 128, 64}));
  AddShape(builder, "left_shape", {12, 32, 128});
  AddShape(builder, "right_shape", {12, 128, 64});
  AddShape(builder, "out_shape", {2, 6, 32, 64});
  builder.MakeNode("Reshape", {"left", "left_shape"}, {"left3"});
  builder.MakeNode("Reshape", {"right", "right_shape"}, {"right3"});
  builder.MakeNode("MatMul", {"left3", "right3"}, {"mm"});
  builder.MakeNode("Reshape", {"mm", "out_shape"}, {"out"});
  SeedShape(builder, "left3", {12, 32, 128});
  SeedShape(builder, "right3", {12, 128, 64});
  SeedShape(builder, "mm", {12, 32, 64});
  SeedShape(builder, "out", {2, 6, 32, 64});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReshapeMatMulReshapePattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[2]).pattern, nullptr);
}

TEST(MulMulMatMulPattern, RejectsGraphOutputIntermediate) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({32, 16}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({16, 64}));
  AddFloat(builder, "c1", {}, {0.4F});
  AddFloat(builder, "c2", {}, {0.6F});
  builder.MakeNode("Mul", {"x", "c1"}, {"left"});
  builder.MakeNode("Mul", {"y", "c2"}, {"right"});
  builder.MakeNode("MatMul", {"left", "right"}, {"out"});
  builder.MakeOutput("left", core::symbolic::TensorType::kFloat, Shape({32, 16}));
  core::builder::GraphGraph graph(builder);
  onnx_patterns::MulMulMatMulPattern pattern;
  EXPECT_TRUE(graph.IsOutput("left"));
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[2]).pattern, nullptr);
}

TEST(SwitchReshapeActivationPattern, RejectsPReluAndSoftmax) {
  onnx_patterns::SwitchReshapeActivationPattern pattern;

  core::builder::GraphBuilder prelu("prelu", SchemaLookup());
  prelu.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  prelu.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  prelu.MakeInput("slope", core::symbolic::TensorType::kFloat, Shape({2}));
  prelu.MakeNode("MatMul", {"x", "w"}, {"mm"});
  prelu.MakeNode("Transpose", {"mm"}, {"layout"}, "", "", PermAttr({1, 0}));
  prelu.MakeNode("PRelu", {"layout", "slope"}, {"out"});
  core::builder::GraphGraph prelu_graph(prelu);
  EXPECT_EQ(pattern.Match(prelu_graph, prelu.Nodes()[2]).pattern, nullptr);

  core::builder::GraphBuilder softmax("softmax", SchemaLookup());
  softmax.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  softmax.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  softmax.MakeNode("MatMul", {"x", "w"}, {"mm"});
  softmax.MakeNode("Transpose", {"mm"}, {"layout"}, "", "", PermAttr({1, 0}));
  softmax.MakeNode("Softmax", {"layout"}, {"out"}, "", "", IntAttr("axis", 0));
  core::builder::GraphGraph softmax_graph(softmax);
  EXPECT_EQ(pattern.Match(softmax_graph, softmax.Nodes()[2]).pattern, nullptr);
}

TEST(ShapeBasedMatMulToMulPattern, RejectsRankTwoTransposeRewrite) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3, 1}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({1, 4}));
  builder.MakeNode("MatMul", {"x", "y"}, {"mm"});
  builder.MakeNode("Transpose", {"mm"}, {"out"}, "", "", PermAttr({1, 0}));
  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedMatMulToMulPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(TransposeMatMulPattern, RejectsTwoInputGemmBeforeOpset11) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 10);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({128, 32}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({128, 64}));
  builder.MakeNode("Transpose", {"x"}, {"xt"}, "", "", PermAttr({1, 0}));
  builder.MakeNode("MatMul", {"xt", "y"}, {"out"});
  SeedShape(builder, "xt", {32, 128});
  core::builder::GraphGraph graph(builder);
  onnx_patterns::TransposeMatMulPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(MatMulPatterns, RejectsForeignDomainAdjacentNodes) {
  core::builder::GraphBuilder add("add", SchemaLookup());
  add.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  add.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  add.MakeInput("bias", core::symbolic::TensorType::kFloat, Shape({4}));
  add.MakeNode("MatMul", {"x", "w"}, {"mm"});
  add.MakeNode("Add", {"mm", "bias"}, {"out"});
  SetNodeDomain(add, 1, "custom.domain");
  core::builder::GraphGraph add_graph(add);
  onnx_patterns::MatMulAddPattern add_pattern;
  EXPECT_EQ(add_pattern.Match(add_graph, add.Nodes()[0]).pattern, nullptr);

  core::builder::GraphBuilder mul("mul", SchemaLookup());
  mul.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  mul.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  AddFloat(mul, "c1", {}, {2.0F});
  AddFloat(mul, "c2", {}, {3.0F});
  mul.MakeNode("Mul", {"x", "c1"}, {"left"});
  mul.MakeNode("Mul", {"y", "c2"}, {"right"});
  mul.MakeNode("MatMul", {"left", "right"}, {"out"});
  SetNodeDomain(mul, 0, "custom.domain");
  core::builder::GraphGraph mul_graph(mul);
  onnx_patterns::MulMulMatMulPattern mul_pattern;
  EXPECT_EQ(mul_pattern.Match(mul_graph, mul.Nodes()[2]).pattern, nullptr);

  core::builder::GraphBuilder layout("layout", SchemaLookup());
  layout.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  layout.MakeInput("w", core::symbolic::TensorType::kFloat, Shape({3, 4}));
  layout.MakeNode("MatMul", {"x", "w"}, {"mm"});
  layout.MakeNode("Transpose", {"mm"}, {"transposed"}, "", "", PermAttr({1, 0}));
  layout.MakeNode("Relu", {"transposed"}, {"out"});
  SetNodeDomain(layout, 1, "custom.domain");
  core::builder::GraphGraph layout_graph(layout);
  onnx_patterns::SwitchReshapeActivationPattern layout_pattern;
  EXPECT_EQ(layout_pattern.Match(layout_graph, layout.Nodes()[2]).pattern, nullptr);

  core::builder::GraphBuilder shape("shape", SchemaLookup());
  shape.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3, 1}));
  shape.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 1, 4}));
  shape.MakeNode("MatMul", {"x", "y"}, {"mm"});
  shape.MakeNode("Transpose", {"mm"}, {"out"}, "", "", PermAttr({0, 2, 1}));
  SetNodeDomain(shape, 1, "custom.domain");
  core::builder::GraphGraph shape_graph(shape);
  onnx_patterns::ShapeBasedMatMulToMulPattern shape_pattern;
  const auto shape_match = shape_pattern.Match(shape_graph, shape.Nodes()[0]);
  ASSERT_EQ(shape_match.pattern, &shape_pattern);
  ASSERT_EQ(shape_match.nodes.size(), 2u);
  EXPECT_EQ(shape_match.nodes[1], nullptr);
}

} // namespace
} // namespace Test
