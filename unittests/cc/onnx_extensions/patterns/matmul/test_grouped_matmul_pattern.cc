// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/dispatch_table.h"
#include "onnx_extensions/patterns/matmul/matmul_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

using core::builder::GraphBuilder;
using core::builder::GraphGraph;
using core::symbolic::TensorType;

GraphBuilder::SchemaLookupFn Schemas() {
  return [](const std::string &op) { return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false); };
}

core::symbolic::SymShape Dims(std::initializer_list<int64_t> dims) {
  core::symbolic::SymShape shape;
  for (int64_t dim : dims) {
    shape.PushBack(core::symbolic::SymDim(dim));
  }
  return shape;
}

utils::RepeatedProtoField<NodeProto> &Nodes(GraphBuilder &builder) {
  return const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
}

NodeProto &Last(GraphBuilder &builder) { return Nodes(builder)[builder.Nodes().size() - 1]; }

void SetDescriptor(GraphBuilder &builder, const std::string &name, TensorType type,
                   std::initializer_list<int64_t> dims) {
  auto &shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
  shapes.Set(name, core::symbolic::SymTensor(nullptr, type, Dims(dims)));
}

void Inputs(GraphBuilder &builder, int64_t tokens, TensorType dtype, TensorType itype, bool bias) {
  builder.MakeInput("x", dtype, Dims({tokens, 3}));
  builder.MakeInput("w", dtype, Dims({4, 3, 5}));
  builder.MakeInput("indices", itype, Dims({tokens, 2}));
  if (bias) {
    builder.MakeInput("bias", dtype, Dims({4, 5}));
  }
}

std::string Row(GraphBuilder &builder, const std::string &token, const std::string &prefix,
                bool bias, bool reverse) {
  builder.MakeNode("Gather", {"x", token}, {prefix + "x"});
  builder.MakeNode("Gather", {"indices", token}, {prefix + "i"});
  builder.MakeNode("Gather", {"w", prefix + "i"}, {prefix + "w"});
  builder.MakeNode("MatMul", {prefix + "x", prefix + "w"}, {prefix + "mm"});
  if (!bias) {
    return prefix + "mm";
  }
  builder.MakeNode("Gather", {"bias", prefix + "i"}, {prefix + "b"});
  builder.MakeNode("Add",
                   reverse ? std::vector<std::string>{prefix + "b", prefix + "mm"}
                           : std::vector<std::string>{prefix + "mm", prefix + "b"},
                   {prefix + "sum"});
  return prefix + "sum";
}

void Concat(GraphBuilder &builder, bool bias = false, bool reverse = false,
            TensorType dtype = TensorType::kFloat, TensorType itype = TensorType::kInt64,
            int tokens = 2) {
  Inputs(builder, tokens, dtype, itype, bias);
  builder.MakeInitializer(MakeInitializerShape("axes", {0}));
  std::vector<std::string> rows;
  for (int m = 0; m < tokens; ++m) {
    const std::string prefix = "r" + std::to_string(m);
    builder.MakeInitializer(MakeInitializer<int64_t>((prefix + "token").c_str(), {}, {m}));
    const std::string row = Row(builder, prefix + "token", prefix, bias, reverse);
    builder.MakeNode("Unsqueeze", {row, "axes"}, {prefix + "out"});
    rows.push_back(prefix + "out");
  }
  const auto concat = MakeNode("Concat", rows, {"out"});
  auto attributes = concat.attribute();
  auto &axis = attributes.add();
  axis.set_name("axis");
  axis.set_type(AttributeProto::INT);
  axis.set_i(0);
  builder.MakeNode("Concat", rows, {"out"}, "", "concat", attributes);
  Last(builder).set_doc_string("expert output");
  auto *metadata = Last(builder).add_metadata_props();
  metadata->set_key("description");
  metadata->set_value("per-token experts");
  builder.MakeOutput("out");
}

void Loop(GraphBuilder &builder, bool bias = false, bool reverse = false,
          TensorType dtype = TensorType::kFloat, TensorType itype = TensorType::kInt64,
          int64_t tokens = 2, bool identity = true, bool omitted = false) {
  Inputs(builder, tokens, dtype, itype, bias);
  builder.MakeInitializer(MakeInitializer<int64_t>("trip", {}, {tokens}));
  auto condition = MakeInitializer<int32_t>("true", {}, {1});
  condition.set_data_type(TensorProto::BOOL);
  builder.MakeInitializer(condition);
  auto &body = builder.MakeSubgraph("body");
  body.MakeInput("iteration", TensorType::kInt64, Dims({}));
  body.MakeInput("condition", TensorType::kBool, Dims({}));
  const std::string row = Row(body, "iteration", "row", bias, reverse);
  if (identity) {
    body.MakeNode("Identity", {"condition"}, {"next_condition"});
  }
  body.MakeOutput(identity ? "next_condition" : "condition");
  body.MakeOutput(row);
  utils::RepeatedProtoField<AttributeProto> attributes;
  auto &attr = attributes.add();
  attr.set_name("body_ref");
  attr.set_type(AttributeProto::STRING);
  attr.set_s("body");
  builder.MakeNode("Loop", {"trip", omitted ? "" : "true"}, {"out"}, "", "loop", attributes);
  Last(builder).set_doc_string("expert output");
  auto *metadata = Last(builder).add_metadata_props();
  metadata->set_key("description");
  metadata->set_value("per-token experts");
  builder.MakeOutput("out");
}

void Squeeze(GraphBuilder &builder, bool bias) {
  Inputs(builder, 2, TensorType::kFloat, TensorType::kInt64, bias);
  builder.MakeInitializer(MakeInitializerShape("unsqueeze_axes", {1, 2}));
  builder.MakeInitializer(MakeInitializerShape("squeeze_axes", {2}));
  builder.MakeNode("Unsqueeze", {"x", "unsqueeze_axes"}, {"xu"});
  builder.MakeNode("Gather", {"w", "indices"}, {"selected"});
  builder.MakeNode("MatMul", {"xu", "selected"}, {"mm"});
  builder.MakeNode("Squeeze", {"mm", "squeeze_axes"}, {bias ? "squeezed" : "out"});
  if (bias) {
    builder.MakeNode("Gather", {"bias", "indices"}, {"selected_bias"});
    builder.MakeNode("Add", {"squeezed", "selected_bias"}, {"out"});
  }
  builder.MakeOutput("out");
}

void CheckReplacement(GraphBuilder &builder, const utils::RepeatedProtoField<NodeProto> &nodes,
                      bool bias, TensorType itype) {
  ASSERT_EQ(nodes.size(), 4u);
  EXPECT_EQ(nodes[0].op_type().value(), "Less");
  EXPECT_EQ(nodes[1].op_type().value(), "Where");
  EXPECT_EQ(nodes[2].op_type().value(), "Add");
  EXPECT_EQ(nodes[0].input()[0].value(), "indices");
  EXPECT_EQ(nodes[1].input()[0], nodes[0].output()[0]);
  EXPECT_EQ(nodes[1].input()[2], nodes[0].input()[1]);
  EXPECT_EQ(nodes[2].input()[0].value(), "indices");
  EXPECT_EQ(nodes[2].input()[1], nodes[1].output()[0]);
  const auto &fused = nodes[3];
  EXPECT_EQ(fused.op_type().value(), "GroupedMatMul");
  EXPECT_TRUE(fused.domain().value().empty());
  EXPECT_EQ(fused.input_size(), bias ? 4 : 3);
  EXPECT_EQ(fused.input()[0].value(), "x");
  EXPECT_EQ(fused.input()[1].value(), "w");
  EXPECT_EQ(fused.input()[2], nodes[2].output()[0]);
  if (bias) {
    EXPECT_EQ(fused.input()[3].value(), "bias");
  }
  EXPECT_EQ(fused.output()[0].value(), "out");
  EXPECT_EQ(fused.doc_string().value(), "expert output");
  EXPECT_EQ(fused.attribute_size(), 0);
  EXPECT_TRUE(fused.overload().value().empty());
  EXPECT_EQ(fused.device_configurations().size(), 0u);
  ASSERT_EQ(fused.metadata_props().size(), 1u);
  EXPECT_EQ(fused.metadata_props()[0].key().value(), "description");
  EXPECT_EQ(fused.metadata_props()[0].value().value(), "per-token experts");
  for (const auto &initializer : builder.Initializers()) {
    if (initializer.name().value() == nodes[1].input()[1].value() ||
        initializer.name().value() == nodes[0].input()[1].value()) {
      EXPECT_EQ(initializer.data_type(), core::symbolic::TensorTypeToDataType(itype));
      std::vector<int64_t> values;
      ASSERT_TRUE(ReadIntegerValues(initializer, values));
      EXPECT_EQ(values, std::vector<int64_t>{
                            initializer.name().value() == nodes[1].input()[1].value() ? 4 : 0});
    }
  }
}

TEST(GroupedMatMulPattern, ConcatSupportedTypesBiasOrdersAndTokenConstants) {
  for (auto dtype :
       {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16}) {
    for (auto itype : {TensorType::kInt32, TensorType::kInt64}) {
      for (int form = 0; form < 3; ++form) {
        GraphBuilder builder("g", Schemas(), {{"", 29}});
        Concat(builder, form != 0, form == 2, dtype, itype, 3);
        GraphGraph graph(builder);
        onnx_patterns::GroupedMatMulPattern pattern;
        EXPECT_EQ(pattern.Name(), "GroupedMatMul");
        EXPECT_EQ(pattern.FastOpType(), (std::set<std::string>{"Concat", "Loop", "Squeeze"}));
        const auto match = pattern.Match(graph, Last(builder));
        ASSERT_EQ(match.pattern, &pattern);
        CheckReplacement(builder, pattern.Apply(graph, match.nodes), form != 0, itype);
      }
    }
  }
}

TEST(GroupedMatMulPattern, LoopSupportedTypesBiasOrdersAndConditions) {
  for (auto dtype :
       {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16}) {
    for (auto itype : {TensorType::kInt32, TensorType::kInt64}) {
      for (int form = 0; form < 3; ++form) {
        for (int condition = 0; condition < 4; ++condition) {
          GraphBuilder builder("g", Schemas(), {{"", 29}});
          Loop(builder, form != 0, form == 2, dtype, itype, 2, condition == 0, condition >= 2);
          if (condition == 3) {
            Last(builder).clear_input();
            Last(builder).add_input("trip");
          }
          GraphGraph graph(builder);
          onnx_patterns::GroupedMatMulPattern pattern;
          EXPECT_EQ(pattern.Name(), "GroupedMatMul");
          EXPECT_EQ(pattern.FastOpType(), (std::set<std::string>{"Concat", "Loop", "Squeeze"}));
          const auto match = pattern.Match(graph, Last(builder));
          ASSERT_EQ(match.pattern, &pattern) << condition;
          CheckReplacement(builder, pattern.Apply(graph, match.nodes), form != 0, itype);
        }
      }
    }
  }
}

TEST(GroupedMatMulPatterns, RegistrationCleanupAndConvergence) {
  for (bool loop : {false, true}) {
    for (bool bias : {false, true}) {
      GraphBuilder builder("g", Schemas(), {{"", 29}});
      if (loop) {
        Loop(builder, bias);
      } else {
        Concat(builder, bias);
      }
      std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
      patterns.push_back(onnx_patterns::CreatePattern("GroupedMatMul"));
      GraphGraph graph(builder, std::move(patterns));
      EXPECT_FALSE(graph.Optimize().rewrites().empty());
      ASSERT_EQ(builder.Nodes().size(), 4u);
      EXPECT_EQ(Last(builder).op_type().value(), "GroupedMatMul");
      EXPECT_EQ(builder.GetShape("out").Shape(), Dims({2, 2, 5}));
      EXPECT_EQ(Last(builder).input_size(), bias ? 4 : 3);
      EXPECT_TRUE(graph.Optimize().rewrites().empty());
      EXPECT_EQ(builder.ToModel().graph().node().size(), 4u);
    }
  }
}

TEST(GroupedMatMulPatterns, SameInstanceAllFormsAndDefaultOptimizerConvergence) {
  onnx_patterns::RegisterPatterns();
  onnx_patterns::GroupedMatMulPattern pattern;
  EXPECT_EQ(pattern.priority, -1);
  for (int form = 0; form < 3; ++form) {
    for (bool bias : {false, true}) {
      GraphBuilder builder("g", Schemas(), {{"", 29}});
      if (form == 0) {
        Squeeze(builder, bias);
      } else if (form == 1) {
        Loop(builder, bias);
      } else {
        Concat(builder, bias);
      }
      GraphGraph graph(builder);
      const NodeProto *root = form == 0 && bias ? graph.NodeBefore("squeezed") : &Last(builder);
      ASSERT_NE(root, nullptr);
      const auto match = pattern.Match(graph, *root);
      ASSERT_EQ(match.pattern, &pattern) << form << bias;
      const auto replacements = pattern.Apply(graph, match.nodes);
      ASSERT_EQ(replacements.size(), 4u);
      EXPECT_EQ(replacements[3].op_type().value(), "GroupedMatMul");
      EXPECT_EQ(replacements[3].name().value().find("GroupedMatMulPattern--"), 0u);
      EXPECT_EQ(replacements[3].output()[0].value(), "out");
      auto wrong_nodes = match.nodes;
      wrong_nodes.push_back(root);
      EXPECT_THROW(pattern.Apply(graph, wrong_nodes), core::builder::BuilderError);
      EXPECT_FALSE(graph.Optimize().rewrites().empty());
      int grouped = 0;
      for (const auto &node : builder.Nodes()) {
        if (node.op_type().value() == "GroupedMatMul") {
          ++grouped;
          EXPECT_EQ(node.input()[0].value(), "x");
          EXPECT_EQ(node.input()[1].value(), "w");
          EXPECT_EQ(node.input_size(), bias ? 4 : 3);
          EXPECT_EQ(node.output()[0].value(), "out");
        }
      }
      EXPECT_EQ(grouped, 1) << form << bias;
      EXPECT_EQ(builder.GetShape("out").Shape(), Dims({2, 2, 5}));
      EXPECT_TRUE(graph.Optimize().rewrites().empty());
    }
  }
}

TEST(GroupedMatMulPattern, ConcatRejectsReorderedMissingDuplicateAndInconsistentRows) {
  for (int guard = 0; guard < 9; ++guard) {
    GraphBuilder builder("g", Schemas(), {{"", 29}});
    Concat(builder, true);
    auto &nodes = Nodes(builder);
    if (guard == 0) {
      std::swap(*Last(builder).mutable_input(0), *Last(builder).mutable_input(1));
    } else if (guard == 1) {
      Last(builder).clear_input();
      Last(builder).add_input("r0out");
    } else if (guard == 2) {
      *Last(builder).mutable_input(1) = utils::String("r0out");
    } else if (guard == 3) {
      *nodes[7].mutable_input(1) = utils::String("r0token");
    } else if (guard == 4) {
      *nodes[8].mutable_input(1) = utils::String("r0token");
    } else if (guard == 5) {
      builder.MakeInput("other", TensorType::kFloat, Dims({2, 3}));
      *nodes[7].mutable_input(0) = utils::String("other");
    } else if (guard == 6) {
      builder.MakeInput("other_bias", TensorType::kFloat, Dims({4, 5}));
      *nodes[11].mutable_input(0) = utils::String("other_bias");
    } else if (guard == 7) {
      builder.MakeInput("dynamic_row", TensorType::kInt64, Dims({}));
      *nodes[0].mutable_input(1) = utils::String("dynamic_row");
      *nodes[1].mutable_input(1) = utils::String("dynamic_row");
    } else {
      SetDescriptor(builder, "r0token", TensorType::kInt64, {1});
    }
    GraphGraph graph(builder);
    onnx_patterns::GroupedMatMulPattern pattern;
    EXPECT_EQ(pattern.Match(graph, Last(builder)).pattern, nullptr) << guard;
  }
}

TEST(GroupedMatMulPattern, ConcatRejectsSharedExportedAndCapturedIntermediates) {
  for (const std::string value : {"r0x", "r0i", "r0w", "r0mm", "r0b", "r0sum", "r0out"}) {
    for (int use = 0; use < 3; ++use) {
      GraphBuilder builder("g", Schemas(), {{"", 29}});
      Concat(builder, true);
      const std::size_t concat = builder.Nodes().size() - 1;
      if (use == 0) {
        builder.MakeNode("Identity", {value}, {"other"});
      } else if (use == 1) {
        builder.MakeOutput(value);
      } else {
        auto &body = builder.MakeSubgraph("capture");
        body.MakeNode("Identity", {value}, {"captured"});
        body.MakeOutput("captured");
        utils::RepeatedProtoField<AttributeProto> attrs;
        auto &attr = attrs.add();
        attr.set_name("body_ref");
        attr.set_type(AttributeProto::STRING);
        attr.set_s("capture");
        builder.MakeNode("Carrier", {}, {"other"}, "", "", attrs);
      }
      GraphGraph graph(builder);
      onnx_patterns::GroupedMatMulPattern pattern;
      EXPECT_EQ(pattern.Match(graph, builder.Nodes()[concat]).pattern, nullptr) << value << use;
    }
  }
}

TEST(GroupedMatMulPatterns, RejectsMetadataAndOpsets) {
  for (bool loop : {false, true}) {
    for (int guard = 0; guard < 12; ++guard) {
      GraphBuilder builder("g", Schemas(), {{"", guard == 0 ? 28 : 29}});
      if (loop) {
        Loop(builder, true);
      } else {
        Concat(builder, true);
      }
      if (guard == 1) {
        SetDescriptor(builder, "x", TensorType::kFloat, {1, 3});
      } else if (guard == 2) {
        SetDescriptor(builder, "x", TensorType::kFloat, {2, 7});
      } else if (guard == 3) {
        SetDescriptor(builder, "w", TensorType::kFloat, {0, 3, 5});
      } else if (guard == 4) {
        SetDescriptor(builder, "indices", TensorType::kFloat, {2, 2});
      } else if (guard == 5) {
        SetDescriptor(builder, "w", TensorType::kDouble, {4, 3, 5});
      } else if (guard == 6) {
        SetDescriptor(builder, "bias", TensorType::kFloat, {1, 5});
      } else if (guard == 7) {
        SetDescriptor(builder, "bias", TensorType::kDouble, {4, 5});
      } else if (guard == 8) {
        SetDescriptor(builder, "indices", TensorType::kInt32, {2, 2});
        SetDescriptor(builder, "w", TensorType::kFloat,
                      {int64_t{std::numeric_limits<int32_t>::max()} + 1, 3, 5});
      } else if (guard == 9) {
        SetDescriptor(builder, "x", TensorType::kInt64, {2, 3});
      } else if (guard == 10) {
        SetDescriptor(builder, "indices", TensorType::kInt64, {2});
      } else if (guard == 11) {
        SetDescriptor(builder, "w", TensorType::kFloat, {4, 3});
      }
      GraphGraph graph(builder);
      const auto pattern = onnx_patterns::CreatePattern("GroupedMatMul");
      EXPECT_EQ(pattern->Match(graph, Last(builder)).pattern, nullptr) << loop << guard;
    }
  }
}

TEST(GroupedMatMulPatterns, RejectsForeignDomainsAndAxes) {
  for (bool loop : {false, true}) {
    const int count = loop ? 7 : 15;
    for (int index = 0; index < count; ++index) {
      GraphBuilder builder("g", Schemas(), {{"", 29}});
      if (loop) {
        Loop(builder, true);
      } else {
        Concat(builder, true);
      }
      auto &nodes = loop ? Nodes(builder.Subgraph("body")) : Nodes(builder);
      nodes[index].set_domain("foreign");
      GraphGraph graph(builder);
      const auto pattern = onnx_patterns::CreatePattern("GroupedMatMul");
      EXPECT_EQ(pattern->Match(graph, Last(builder)).pattern, nullptr) << loop << index;
    }
    for (int index : {0, 1, 2, 4}) {
      GraphBuilder builder("g", Schemas(), {{"", 29}});
      if (loop) {
        Loop(builder, true);
      } else {
        Concat(builder, true);
      }
      auto &nodes = loop ? Nodes(builder.Subgraph("body")) : Nodes(builder);
      AddAttribute<int64_t>(nodes[index], "axis", 1);
      GraphGraph graph(builder);
      const auto pattern = onnx_patterns::CreatePattern("GroupedMatMul");
      EXPECT_EQ(pattern->Match(graph, Last(builder)).pattern, nullptr);
    }
  }
}

TEST(GroupedMatMulPatterns, RejectsOverloadedOperators) {
  for (bool loop : {false, true}) {
    for (int index = 0; index < (loop ? 8 : 15); ++index) {
      GraphBuilder builder("g", Schemas(), {{"", 29}});
      if (loop) {
        Loop(builder, true);
      } else {
        Concat(builder, true);
      }
      NodeProto &node = loop && index < 7 ? Nodes(builder.Subgraph("body"))[index]
                                          : (loop ? Last(builder) : Nodes(builder)[index]);
      node.set_overload("model_local");
      GraphGraph graph(builder);
      const auto pattern = onnx_patterns::CreatePattern("GroupedMatMul");
      EXPECT_EQ(pattern->Match(graph, Last(builder)).pattern, nullptr) << loop << index;
      EXPECT_THROW(pattern->Apply(graph, {&Last(builder)}), core::builder::BuilderError);
    }
  }
}

TEST(GroupedMatMulPattern, LoopRejectsTripConditionsStateAndExtraOperations) {
  for (int guard = 0; guard < 15; ++guard) {
    GraphBuilder builder("g", Schemas(), {{"", 29}});
    Loop(builder, true);
    auto &loop = Last(builder);
    auto &body = builder.Subgraph("body");
    if (guard == 0) {
      builder.MakeInput("dynamic_trip", TensorType::kInt64, Dims({}));
      *loop.mutable_input(0) = utils::String("dynamic_trip");
    } else if (guard == 1) {
      builder.MakeInitializer(MakeInitializer<int64_t>("wrong_trip", {}, {1}));
      *loop.mutable_input(0) = utils::String("wrong_trip");
    } else if (guard == 2) {
      *loop.mutable_input(0) = utils::String("");
    } else if (guard == 3) {
      builder.MakeInput("dynamic_condition", TensorType::kBool, Dims({}));
      *loop.mutable_input(1) = utils::String("dynamic_condition");
    } else if (guard == 4) {
      auto condition = MakeInitializer<int32_t>("false", {}, {0});
      condition.set_data_type(TensorProto::BOOL);
      builder.MakeInitializer(condition);
      *loop.mutable_input(1) = utils::String("false");
    } else if (guard == 5) {
      loop.add_input("x");
    } else if (guard == 6) {
      loop.add_output("another_scan");
    } else if (guard == 7) {
      Last(body).set_op_type("Not");
    } else if (guard == 8) {
      body.MakeNode("Neg", {"rowmm"}, {"unused"});
    } else if (guard == 9) {
      body.MakeSubgraph("nested");
    } else if (guard == 10) {
      *Nodes(body)[0].mutable_input(1) = utils::String("condition");
    } else if (guard == 11) {
      SetDescriptor(body, "iteration", TensorType::kInt32, {});
    } else if (guard == 12) {
      SetDescriptor(body, "condition", TensorType::kBool, {1});
    } else if (guard == 13) {
      SetDescriptor(builder, "trip", TensorType::kInt64, {1});
    } else {
      loop.set_domain("foreign");
    }
    GraphGraph graph(builder);
    onnx_patterns::GroupedMatMulPattern pattern;
    EXPECT_EQ(pattern.Match(graph, loop).pattern, nullptr) << guard;
    EXPECT_THROW(pattern.Apply(graph, {&loop}), core::builder::BuilderError);
  }
}

TEST(GroupedMatMulPattern, LoopRejectsShadowedCaptures) {
  for (const std::string capture : {"x", "w", "indices", "bias"}) {
    for (int local = 0; local < 3; ++local) {
      GraphBuilder builder("g", Schemas(), {{"", 29}});
      Loop(builder, true);
      auto &body = builder.Subgraph("body");
      if (local == 0) {
        // A local initializer may shadow an outer capture without changing its shape.
        auto &initializers =
            const_cast<utils::RepeatedProtoField<TensorProto> &>(body.Initializers());
        initializers.push_back(MakeInitializer<float>(capture.c_str(), {1}, {0}));
      } else if (local == 1) {
        auto &inputs = const_cast<utils::RepeatedProtoField<ValueInfoProto> &>(body.Inputs());
        inputs[0].set_name(capture);
        *Nodes(body)[0].mutable_input(1) = utils::String(capture);
        *Nodes(body)[1].mutable_input(1) = utils::String(capture);
      } else {
        // Mutates an existing local producer rather than adding an extra operation.
        *Nodes(body)[0].mutable_output(0) = utils::String(capture);
        *Nodes(body)[3].mutable_input(0) = utils::String(capture);
      }
      GraphGraph graph(builder);
      onnx_patterns::GroupedMatMulPattern pattern;
      EXPECT_EQ(pattern.Match(graph, Last(builder)).pattern, nullptr) << capture << local;
    }
  }
}

TEST(GroupedMatMulPattern, LoopRejectsSparseInitializerCaptureShadowing) {
  for (const std::string capture : {"x", "w", "indices", "bias"}) {
    GraphBuilder source("g", Schemas(), {{"", 29}});
    Loop(source, true);
    ModelProto model = source.ToModel();
    auto *body = model.mutable_graph()->mutable_node(0)->mutable_attribute(0)->mutable_g();
    auto *sparse = body->add_sparse_initializer();
    *sparse->mutable_values() = capture == "indices"
                                    ? MakeInitializer<int64_t>(capture.c_str(), {1}, {0})
                                    : MakeInitializer<float>(capture.c_str(), {1}, {1});
    *sparse->mutable_indices() = MakeInitializer<int64_t>("", {1}, {0});
    const std::vector<int64_t> dims = capture == "x"      ? std::vector<int64_t>{2, 3}
                                      : capture == "w"    ? std::vector<int64_t>{4, 3, 5}
                                      : capture == "bias" ? std::vector<int64_t>{4, 5}
                                                          : std::vector<int64_t>{2, 2};
    for (int64_t dim : dims) {
      sparse->dims().push_back(dim);
    }
    GraphBuilder builder(model, Schemas());
    const auto &imported_body = builder.Subgraph(Last(builder).attribute(0).s().value());
    std::unordered_set<std::string> captures;
    imported_body.CollectImplicitInputs(captures);
    EXPECT_EQ(captures.count(capture), 0u);
    GraphGraph graph(builder);
    onnx_patterns::GroupedMatMulPattern pattern;
    EXPECT_EQ(pattern.Match(graph, Last(builder)).pattern, nullptr) << capture;
    EXPECT_THROW(pattern.Apply(graph, {&Last(builder)}), core::builder::BuilderError);
  }
}

TEST(GroupedMatMulPatterns, TokenBoundariesAndUnsafeApply) {
  for (int tokens : {0, 1, 2}) {
    GraphBuilder builder("g", Schemas(), {{"", 29}});
    // A zero-token Concat is not a valid ONNX graph; use one branch with M=0.
    Concat(builder, false, false, TensorType::kFloat, TensorType::kInt64, tokens == 0 ? 1 : tokens);
    if (tokens == 0) {
      SetDescriptor(builder, "x", TensorType::kFloat, {0, 3});
      SetDescriptor(builder, "indices", TensorType::kInt64, {0, 2});
    }
    GraphGraph graph(builder);
    onnx_patterns::GroupedMatMulPattern pattern;
    auto match = pattern.Match(graph, Last(builder));
    EXPECT_EQ(match.pattern != nullptr, tokens >= 2);
    EXPECT_THROW(pattern.Apply(graph, {}), core::builder::BuilderError);
    if (match.pattern) {
      std::swap(match.nodes[1], match.nodes[2]);
      EXPECT_THROW(pattern.Apply(graph, match.nodes), core::builder::BuilderError);
    }
  }
  for (int tokens : {0, 1, 2}) {
    GraphBuilder builder("g", Schemas(), {{"", 29}});
    Loop(builder, false, false, TensorType::kFloat, TensorType::kInt64, tokens);
    GraphGraph graph(builder);
    onnx_patterns::GroupedMatMulPattern pattern;
    const auto match = pattern.Match(graph, Last(builder));
    EXPECT_EQ(match.pattern != nullptr, tokens > 0);
    EXPECT_THROW(pattern.Apply(graph, {}), core::builder::BuilderError);
    EXPECT_THROW(pattern.Apply(graph, {&Last(builder), &Last(builder)}),
                 core::builder::BuilderError);
  }
}

TEST(GroupedMatMulPattern, ConcatSupportsInt32ScalarRows) {
  GraphBuilder builder("g", Schemas(), {{"", 29}});
  Concat(builder);
  for (int m = 0; m < 2; ++m) {
    const auto name = "token32_" + std::to_string(m);
    builder.MakeInitializer(MakeInitializer<int32_t>(name.c_str(), {}, {m}));
    *Nodes(builder)[m * 5].mutable_input(1) = utils::String(name);
    *Nodes(builder)[m * 5 + 1].mutable_input(1) = utils::String(name);
  }
  GraphGraph graph(builder);
  onnx_patterns::GroupedMatMulPattern pattern;
  EXPECT_EQ(pattern.Match(graph, Last(builder)).pattern, &pattern);
}

TEST(GroupedMatMulPattern, ConcatRejectsInvalidUnsqueezeAndConcatAxes) {
  for (int guard = 0; guard < 7; ++guard) {
    GraphBuilder builder("g", Schemas(), {{"", 29}});
    Concat(builder);
    if (guard == 0) {
      Last(builder).clear_attribute();
    } else if (guard == 1) {
      Last(builder).mutable_attribute(0)->set_i(1);
    } else if (guard == 2) {
      builder.MakeInitializer(MakeInitializerShape("wrong_axes", {1}));
      *Nodes(builder)[4].mutable_input(1) = utils::String("wrong_axes");
    } else if (guard == 3) {
      builder.MakeInput("dynamic_axes", TensorType::kInt64, Dims({1}));
      *Nodes(builder)[4].mutable_input(1) = utils::String("dynamic_axes");
    } else if (guard == 4) {
      SetDescriptor(builder, "axes", TensorType::kInt32, {1});
    } else if (guard == 5) {
      SetDescriptor(builder, "axes", TensorType::kInt64, {});
    } else {
      Last(builder).mutable_attribute(0)->set_ref_attr_name("unknown_axis");
    }
    GraphGraph graph(builder);
    onnx_patterns::GroupedMatMulPattern pattern;
    EXPECT_EQ(pattern.Match(graph, Last(builder)).pattern, nullptr) << guard;
  }
}

TEST(GroupedMatMulPatterns, NormalizationKeepsIntegerBoundariesSafe) {
  for (bool loop : {false, true}) {
    for (auto itype : {TensorType::kInt32, TensorType::kInt64}) {
      const int64_t groups = itype == TensorType::kInt32 ? std::numeric_limits<int32_t>::max()
                                                         : std::numeric_limits<int64_t>::max();
      GraphBuilder builder("g", Schemas(), {{"", 29}});
      if (loop) {
        Loop(builder, false, false, TensorType::kFloat, itype);
      } else {
        Concat(builder, false, false, TensorType::kFloat, itype);
      }
      SetDescriptor(builder, "w", TensorType::kFloat, {groups, 3, 5});
      GraphGraph graph(builder);
      const auto pattern = onnx_patterns::CreatePattern("GroupedMatMul");
      const auto match = pattern->Match(graph, Last(builder));
      ASSERT_NE(match.pattern, nullptr);
      const auto replacements = pattern->Apply(graph, match.nodes);
      // Where selects G or zero before Add; adding G eagerly would overflow
      // for the largest nonnegative index, even when the result is not selected.
      EXPECT_EQ(replacements[1].op_type().value(), "Where");
      EXPECT_EQ(replacements[2].input()[1], replacements[1].output()[0]);
      for (const auto &initializer : builder.Initializers()) {
        if (initializer.name().value() == replacements[1].input()[1].value()) {
          std::vector<int64_t> values;
          ASSERT_TRUE(ReadIntegerValues(initializer, values));
          EXPECT_EQ(values, std::vector<int64_t>{groups});
        }
      }
      for (int64_t index :
           {std::numeric_limits<int64_t>::min(), -groups, int64_t{-1}, int64_t{0}, groups}) {
        const int64_t offset = index < 0 ? groups : 0;
        EXPECT_LE(index + offset, groups);
        if (index == -groups) {
          EXPECT_EQ(index + offset, 0);
        } else if (index == groups) {
          EXPECT_EQ(index + offset, index);
        }
      }
    }
  }
}

TEST(GroupedMatMulPattern, LoopComparesLargeTripCountsExactly) {
  GraphBuilder builder("g", Schemas(), {{"", 29}});
  Loop(builder);
  const int64_t tokens = (int64_t{1} << 53) + 1;
  SetDescriptor(builder, "x", TensorType::kFloat, {tokens, 3});
  SetDescriptor(builder, "indices", TensorType::kInt64, {tokens, 2});
  builder.MakeInitializer(MakeInitializer<int64_t>("rounded_trip", {}, {tokens - 1}));
  *Last(builder).mutable_input(0) = utils::String("rounded_trip");
  GraphGraph graph(builder);
  onnx_patterns::GroupedMatMulPattern pattern;
  EXPECT_EQ(pattern.Match(graph, Last(builder)).pattern, nullptr);
}

} // namespace
} // namespace Test
