// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/shapes/dispatch_table.h"

#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_helper.h"
#include "onnx_lib/checker.h"
#include "onnx_manipulations/parser.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_ort_flatbuffers.h"
#include "onnx_proto/onnx_verify.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

// Schema provider backed by the built-in ONNX operator schemas.
core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, /*init_doc=*/false);
  };
}

core::symbolic::SymShape MakeShape(std::initializer_list<int64_t> dims) {
  core::symbolic::SymShape shape;
  for (int64_t d : dims) {
    shape.PushBack(core::symbolic::SymDim(d));
  }
  return shape;
}

} // namespace

TEST(GraphBuilder, StartsEmpty) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  EXPECT_EQ(builder.Nodes().size(), 0u);
  EXPECT_EQ(builder.Inputs().size(), 0u);
  EXPECT_FALSE(builder.HasName("x"));
}

TEST(GraphBuilder, LocalFunctionMemoryIncludesBodyActivationsAndWorkspace) {
  constexpr char domain[] = "function_memory_test";
  core::shapes::RegisterComputeShapeFn(
      domain, "Expand", [](core::shapes::ShapesContext &ctx, const NodeProto &node) {
        ctx.Set(node.output(0),
                core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                          MakeShape({2, 1024})));
      });
  core::shapes::RegisterComputeShapeFn(
      domain, "Project", [](core::shapes::ShapesContext &ctx, const NodeProto &node) {
        ctx.Set(node.output(0),
                core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                          MakeShape({2, 3})));
      });
  core::shapes::RegisterComputePeakMemoryFn(
      domain, "Expand", core::symbolic::Device::kCPU,
      [](core::symbolic::Device, const std::vector<core::symbolic::SymShape> &) {
        return int64_t{1234};
      });

  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}, {"ai.rt", 1}});
  builder.set_device(core::symbolic::Device::kCPU);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  auto &function = builder.MakeLocalFunction("Layer", "ai.rt");
  function.SetOpsetVersion(domain, 1);
  function.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Expand", {"x"}, {"hidden"}, domain);
  function.MakeNode("Project", {"hidden"}, {"result"}, domain);
  function.MakeOutput("result");
  builder.MakeNode("Layer", {"X"}, {"Y"}, "ai.rt");
  builder.MakeOutput("Y");

  const auto total_bytes = [](const core::compute::ComputeContext &context) {
    return std::get<int64_t>(std::get<core::expressions::DimType>(
        context.NodeMemory(0).at(core::compute::kNodeMemoryTotalBytesKey)));
  };
  EXPECT_EQ(total_bytes(builder.Compute()), 8240);
  EXPECT_EQ(builder.Compute().PeakMemory().at(0), 1234);
  EXPECT_EQ(std::get<int64_t>(
                std::get<core::compute::TaggedMemory>(
                    builder.Compute().NodeMemory(0).at(core::compute::kNodeMemoryIntermediatesKey))
                    .at("")),
            8192);

  const ModelProto model = builder.ToModel();
  core::compute::ComputeContext context;
  context.ComputeShapes(model);
  context.ComputeInPlaceReuseGraph(model.graph(), context.Shapes());
  EXPECT_EQ(total_bytes(context), 8240);
  EXPECT_EQ(context.ComputePeakMemory(model.graph(), core::symbolic::Device::kCPU).at(0), 1234);
}

TEST(GraphBuilder, LocalFunctionMemoryKeepsConstantsLiveUntilTheirLastUse) {
  constexpr char domain[] = "function_memory_lifetime_test";
  onnx_shapes::RegisterShapeFunctions();
  core::shapes::RegisterComputeShapeFn(domain, "Copy",
                                       [](core::shapes::ShapesContext &ctx, const NodeProto &node) {
                                         ctx.CopyValueFrom(node.output(0), ctx, node.input(0));
                                       });

  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}, {"ai.rt", 1}});
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({16}));
  auto &function = builder.MakeLocalFunction("ConstantBody", "ai.rt");
  function.SetOpsetVersion(domain, 1);
  function.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({16}));
  utils::RepeatedProtoField<AttributeProto> constant_attributes;
  auto &value = constant_attributes.add();
  value.set_name("value");
  value.set_type(AttributeProto::TENSOR);
  *value.mutable_t() = MakeInitializer<float>("", {8}, std::vector<float>(8, 1.0f));
  function.MakeNode("Constant", {}, {"weight"}, "", "", constant_attributes);
  function.MakeNode("Copy", {"x"}, {"hidden"}, domain);
  function.MakeNode("Copy", {"hidden"}, {"middle"}, domain);
  function.MakeNode("Copy", {"middle", "weight"}, {"result"}, domain);
  function.MakeOutput("result");
  builder.MakeNode("ConstantBody", {"X"}, {"Y"}, "ai.rt");
  builder.MakeOutput("Y");

  const auto total_bytes = [](const core::compute::ComputeContext &context) {
    return std::get<int64_t>(std::get<core::expressions::DimType>(
        context.NodeMemory(0).at(core::compute::kNodeMemoryTotalBytesKey)));
  };
  EXPECT_EQ(total_bytes(builder.Compute()), 288);
  EXPECT_EQ(std::get<int64_t>(
                std::get<core::compute::TaggedMemory>(
                    builder.Compute().NodeMemory(0).at(core::compute::kNodeMemoryIntermediatesKey))
                    .at("")),
            160);

  const ModelProto model = builder.ToModel();
  core::compute::ComputeContext context;
  context.ComputeShapes(model);
  context.ComputeInPlaceReuseGraph(model.graph(), context.Shapes());
  EXPECT_EQ(total_bytes(context), 288);
}

TEST(GraphBuilder, LocalFunctionMemoryDefersRankUnknownTensorInputs) {
  onnx_shapes::RegisterShapeFunctions();
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}, {"custom", 1}});
  auto &function = builder.MakeLocalFunction("Product", "custom");
  for (const char *name : {"A", "B"}) {
    ValueInfoProto input;
    input.set_name(name);
    input.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
    builder.MakeInput(input);
    function.MakeInput(input);
  }
  function.MakeNode("MatMul", {"A", "B"}, {"result"});
  ValueInfoProto output;
  output.set_name("result");
  output.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  function.MakeOutput(output);
  EXPECT_NO_THROW(builder.MakeNode("Product", {"A", "B"}, {"Y"}, "custom"));
  EXPECT_FALSE(builder.Shapes().Has("Y"));

  auto &outer = builder.MakeLocalFunction("Outer", "custom");
  for (const char *name : {"A", "B"}) {
    ValueInfoProto input;
    input.set_name(name);
    input.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
    outer.MakeInput(input);
  }
  outer.MakeNode("Product", {"A", "B"}, {"result"}, "custom");
  outer.MakeOutput(output);

  ValueInfoProto sequence;
  sequence.set_name("sequence");
  sequence.mutable_type()
      ->mutable_sequence_type()
      ->mutable_elem_type()
      ->mutable_tensor_type()
      ->set_elem_type(TensorProto::FLOAT);
  builder.MakeInput(sequence);
  builder.MakeInput("position", core::symbolic::TensorType::kInt64, MakeShape({}));
  builder.MakeNode("SequenceAt", {"sequence", "position"}, {"typed_unknown"});
  ASSERT_TRUE(builder.Shapes().HasType("typed_unknown"));
  EXPECT_EQ(builder.Shapes().GetType("typed_unknown").tensor_type().elem_type(),
            TensorProto::FLOAT);
  ASSERT_FALSE(builder.Shapes().Has("typed_unknown"));
  ASSERT_NO_THROW(
      builder.MakeNode("Outer", {"typed_unknown", "typed_unknown"}, {"nested"}, "custom"));
  EXPECT_FALSE(builder.Shapes().Has("nested"));
  output.set_name("nested");
  builder.MakeOutput(output);

  core::builder::GraphBuilder imported(builder.ToModel(), SchemaLookup());
  ASSERT_TRUE(imported.Shapes().HasType("typed_unknown"));
  EXPECT_EQ(imported.Shapes().GetType("typed_unknown").tensor_type().elem_type(),
            TensorProto::FLOAT);
  EXPECT_FALSE(imported.Shapes().Has("typed_unknown"));
  EXPECT_FALSE(imported.Shapes().Has("nested"));

  builder.MakeInput("scalar", core::symbolic::TensorType::kFloat, MakeShape({}));
  EXPECT_THROW(builder.MakeNode("Outer", {"scalar", "scalar"}, {"invalid"}, "custom"),
               std::invalid_argument);
}

TEST(GraphBuilder, LocalFunctionMemoryMaterializesGraphArgumentsAfterMutations) {
  onnx_shapes::RegisterShapeFunctions();
  constexpr char domain[] = "graph_argument_memory";
  core::shapes::RegisterComputeShapeFn(domain, "Copy",
                                       [](core::shapes::ShapesContext &ctx, const NodeProto &node) {
                                         ctx.CopyValueFrom(node.output(0), ctx, node.input(0));
                                       });
  core::shapes::RegisterComputePeakMemoryFn(
      domain, "Copy", core::symbolic::Device::kCPU,
      [](core::symbolic::Device, const std::vector<core::symbolic::SymShape> &) {
        return int64_t{37};
      });
  core::builder::GraphBuilder body("WithGraph", SchemaLookup(), {{"", 28}, {domain, 1}});
  body.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  body.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  utils::RepeatedProtoField<AttributeProto> references;
  for (const char *name : {"then_branch", "else_branch"}) {
    auto &attribute = references.add();
    attribute.set_name(name);
    attribute.set_ref_attr_name(name);
    attribute.set_type(AttributeProto::GRAPH);
  }
  body.MakeNode("If", {"condition"}, {"selected"}, "", "", references);
  body.MakeNode("Copy", {"selected"}, {"result"}, domain);
  body.MakeOutput("result", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  FunctionProto function = body.ToFunction(domain);
  function.add_attribute("then_branch");
  function.add_attribute("else_branch");

  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}, {domain, 1}});
  builder.set_device(core::symbolic::Device::kCPU);
  builder.MakeLocalFunction(function);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  utils::RepeatedProtoField<AttributeProto> arguments;
  for (const char *name : {"then_branch", "else_branch"}) {
    auto &attribute = arguments.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::GRAPH);
    auto &graph = *attribute.mutable_g();
    graph.set_name(name);
    graph.add_node(MakeNode("Identity", {"X"}, {"branch_result"}));
    graph.add_output()->set_name("branch_result");
  }
  builder.MakeNode("WithGraph", {"X", "condition"}, {"Y"}, domain, "", arguments);
  builder.MakeNode("Identity", {"Y"}, {"Z"});
  builder.MakeOutput("Z");
  EXPECT_EQ(builder.Compute().PeakMemory().at(0), 37);
  const auto &intermediates = std::get<core::compute::TaggedMemory>(
      builder.Compute().NodeMemory(0).at(core::compute::kNodeMemoryIntermediatesKey));
  ASSERT_NE(intermediates.find(""), intermediates.end());
  EXPECT_EQ(std::get<int64_t>(intermediates.at("")), 24);
  const ModelProto model = builder.ToModel();
  core::compute::ComputeContext context;
  context.ComputeShapes(model);
  context.ComputeInPlaceReuseGraph(model.graph(), context.Shapes(), false,
                                   context.ComputeValueAndNodeTags(model.graph()).first);
  EXPECT_EQ(builder.Compute().NodeMemory(0), context.NodeMemory(0));
  EXPECT_EQ(context.ComputePeakMemory(model.graph(), core::symbolic::Device::kCPU).at(0), 37);
}

TEST(GraphBuilder, ToFunctionNormalizesAiOnnxDomainsInNestedGraphAttributes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}, {"custom", 1}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeNode("Identity", {"x"}, {"outer"}, "ai.onnx");

  GraphProto nested;
  nested.set_name("nested");
  nested.add_input()->set_name("nested_x");
  nested.add_node(MakeNode("Identity", {"nested_x"}, {"nested_y"}, "ai.onnx"));
  nested.add_output()->set_name("nested_y");
  utils::RepeatedProtoField<AttributeProto> attributes;
  auto &body = attributes.add();
  body.set_name("body");
  body.set_type(AttributeProto::GRAPH);
  *body.mutable_g() = nested;
  builder.MakeNode("Custom", {"outer"}, {"y"}, "custom", "", attributes);
  builder.MakeOutput("y");

  const FunctionProto function = builder.ToFunction("custom");
  ASSERT_EQ(function.node().size(), 2u);
  EXPECT_EQ(function.node(0).domain(), "");
  EXPECT_EQ(function.node(1).attribute(0).g().node(0).domain(), "");
  for (const auto &opset : function.opset_import())
    EXPECT_NE(opset.domain(), "ai.onnx");
}

TEST(GraphBuilder, CopyDeviceTargetsPreservesReferencedScopesAndFunctions) {
  using core::builder::GraphBuilder;
  using core::symbolic::Device;
  using core::symbolic::TensorType;
  GraphBuilder source("root", SchemaLookup(), {{"", 18}});
  source.MakeInput("x", TensorType::kFloat, MakeShape({2}));
  source.MakeInput("condition", TensorType::kBool, MakeShape({}));
  auto &body = source.MakeSubgraph("body");
  body.MakeNode("Identity", {"x"}, {"result"});
  body.MakeOutput("result");
  body.set_device(Device::kGPU0);
  auto &function = source.MakeLocalFunction("F", "local");
  function.MakeInput("a", TensorType::kFloat, MakeShape({2}));
  function.MakeNode("Neg", {"a"}, {"b"});
  function.MakeOutput("b");
  function.set_device(Device::kCPU);
  body.MakeLocalFunction("Unused", "local");
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto &name : {"then_branch_ref", "else_branch_ref"}) {
    auto &attribute = attributes.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::STRING);
    attribute.set_s("body");
  }
  source.MakeNode("If", {"condition"}, {"y"}, "", "", attributes);
  source.MakeOutput("y");
  GraphBuilder target(source.ToModel(), SchemaLookup());
  target.CopyDeviceTargetsFrom(source);
  EXPECT_EQ(target.device(), Device::kUndefined);
  ASSERT_EQ(target.Subgraphs().size(), 2);
  for (const auto &child : target.Subgraphs())
    EXPECT_EQ(child->device(), Device::kGPU0);
  EXPECT_EQ(target.LocalFunction("F").device(), Device::kCPU);
  EXPECT_EQ(source.device(), Device::kUndefined);
  EXPECT_EQ(body.device(), Device::kGPU0);

  GraphBuilder incompatible("other", SchemaLookup(), {{"", 18}});
  incompatible.set_device(Device::kCPU);
  EXPECT_THROW(incompatible.CopyDeviceTargetsFrom(source), core::builder::BuilderError);
  EXPECT_EQ(incompatible.device(), Device::kCPU);

  target.MakeLocalFunction("Extra", "local");
  target.set_device(Device::kCPU);
  for (const auto &child : target.Subgraphs())
    child->set_device(Device::kCPU);
  EXPECT_THROW(target.CopyDeviceTargetsFrom(source), core::builder::BuilderError);
  EXPECT_EQ(target.device(), Device::kCPU);
  for (const auto &child : target.Subgraphs())
    EXPECT_EQ(child->device(), Device::kCPU);
}

TEST(GraphBuilder, CopyDeviceTargetsAcceptsCompatibleDuplicateFunctionIdentities) {
  using core::builder::GraphBuilder;
  using core::symbolic::Device;
  using core::symbolic::TensorType;
  const auto make_source = [](Device second_function_device) {
    GraphBuilder source("root", SchemaLookup(), {{"", 18}});
    source.MakeInput("condition", TensorType::kBool, MakeShape({}));
    source.MakeInput("x", TensorType::kFloat, MakeShape({2}));
    for (const auto &[scope_name, device] : {std::pair<const char *, Device>{"then", Device::kCPU},
                                             {"else", second_function_device}}) {
      auto &scope = source.MakeSubgraph(scope_name);
      scope.MakeNode("Identity", {"x"}, {"result"});
      scope.MakeOutput("result");
      auto &function = scope.MakeLocalFunction("F", "local");
      function.MakeInput("a", TensorType::kFloat, MakeShape({2}));
      function.MakeNode("Identity", {"a"}, {"b"});
      function.MakeOutput("b");
      function.set_device(device);
    }
    utils::RepeatedProtoField<AttributeProto> attributes;
    for (const auto &[name, scope] :
         {std::pair<const char *, const char *>("then_branch_ref", "then"),
          {"else_branch_ref", "else"}}) {
      auto &attribute = attributes.add();
      attribute.set_name(name);
      attribute.set_type(AttributeProto::STRING);
      attribute.set_s(scope);
    }
    source.MakeNode("If", {"condition"}, {"y"}, "", "", attributes);
    source.MakeOutput("y");
    return source;
  };

  GraphBuilder source = make_source(Device::kCPU);
  GraphBuilder target(source.ToModel(), SchemaLookup());
  ASSERT_NO_THROW(target.CopyDeviceTargetsFrom(source));
  EXPECT_EQ(target.LocalFunction("F").device(), Device::kCPU);

  GraphBuilder incompatible_source = make_source(Device::kGPU0);
  GraphBuilder incompatible_target(incompatible_source.ToModel(), SchemaLookup());
  EXPECT_THROW(incompatible_target.CopyDeviceTargetsFrom(incompatible_source),
               core::builder::BuilderError);
  EXPECT_EQ(incompatible_target.LocalFunction("F").device(), Device::kUndefined);
}

TEST(GraphBuilder, ExportAndDeviceTargetsUseCanonicalFunctionIdentities) {
  using core::builder::GraphBuilder;
  using core::symbolic::Device;
  using core::symbolic::TensorType;
  for (const std::string source_domain : {"", "ai.onnx", "onnx.ai"}) {
    for (const std::string target_domain : {"", "ai.onnx", "onnx.ai"}) {
      for (const std::string overload : {"", "variant:one%"}) {
        SCOPED_TRACE(source_domain + "/" + target_domain + "/" + overload);
        GraphBuilder source("root", SchemaLookup(), {{"", 18}, {"onnx.ai", 18}});
        FunctionProto definition;
        definition.set_name("F");
        definition.set_domain(source_domain);
        definition.set_overload(overload);
        definition.add_opset("", 18);
        definition.add_input("a");
        definition.add_output("b");
        definition.add_node(MakeNode("Neg", {"a"}, {"b"}));
        source.MakeLocalFunction(definition).set_device(Device::kCPU);
        source.MakeInput("x", TensorType::kFloat, MakeShape({2}));
        source.MakeNode("F", {"x"}, {"y"}, source_domain, "",
                        utils::RepeatedProtoField<AttributeProto>(), overload);
        source.MakeOutput("y");

        auto model = source.ToModel();
        ASSERT_EQ(model.functions().size(), 1u);
        model.ref_functions()[0].set_domain(target_domain);
        GraphBuilder target(model, SchemaLookup());
        ASSERT_NO_THROW(target.CopyDeviceTargetsFrom(source));
        EXPECT_EQ(target.LocalFunction("F").device(), Device::kCPU);
        EXPECT_EQ(target.InlineLocalFunctions(), 1u);
        ASSERT_EQ(target.Nodes().size(), 1u);
        EXPECT_EQ(target.Nodes()[0].op_type(), "Neg");

        auto duplicate = model.functions(0);
        duplicate.set_domain(source_domain);
        model.ref_functions().push_back(std::move(duplicate));
        EXPECT_THROW((GraphBuilder(model, SchemaLookup())), core::builder::BuilderError);
        model.ref_functions()[1].set_overload(overload + "different");
        EXPECT_NO_THROW((GraphBuilder(model, SchemaLookup())));
      }
    }
  }
}

TEST(GraphBuilder, CopyDeviceTargetsAlignsDuplicateFunctionBranchesByReference) {
  using core::builder::GraphBuilder;
  using core::symbolic::Device;
  using core::symbolic::TensorType;
  for (const bool compatible : {false, true}) {
    SCOPED_TRACE(compatible);
    GraphBuilder source("root", SchemaLookup(), {{"", 18}});
    source.MakeInput("condition", TensorType::kBool, MakeShape({}));
    source.MakeInput("x", TensorType::kFloat, MakeShape({2}));
    for (const bool reverse : {false, true}) {
      GraphBuilder isolated("isolated", SchemaLookup(), {{"", 18}});
      isolated.MakeInput("x", TensorType::kFloat, MakeShape({2}));
      auto &scope = isolated.MakeSubgraph(reverse ? "right" : "left");
      scope.MakeNode("Identity", {"x"}, {"result"});
      scope.MakeOutput("result");
      auto &function = scope.MakeLocalFunction("F", "local");
      function.MakeInput("flag", TensorType::kBool, MakeShape({}));
      function.MakeInput("a", TensorType::kFloat, MakeShape({2}));
      function.set_device(Device::kCPU);
      for (int index = 0; index < 2; ++index) {
        const bool then_branch = reverse ? index == 1 : index == 0;
        auto &branch = function.MakeSubgraph(then_branch ? "then" : "else");
        branch.MakeNode("Identity", {"a"}, {"out"});
        branch.MakeOutput("out");
        const bool cpu = reverse && !compatible ? !then_branch : then_branch;
        branch.set_device(cpu ? Device::kCPU : Device::kGPU0);
      }
      utils::RepeatedProtoField<AttributeProto> attributes;
      for (const auto &[name, branch] :
           {std::pair<const char *, const char *>{"then_branch_ref", "then"},
            {"else_branch_ref", "else"}}) {
        auto &attribute = attributes.add();
        attribute.set_name(name);
        attribute.set_type(AttributeProto::STRING);
        attribute.set_s(branch);
      }
      function.MakeNode("If", {"flag"}, {"b"}, "", "", attributes);
      function.MakeOutput("b");
      source.MakeSubgraph(scope.name()) = std::move(scope);
    }
    utils::RepeatedProtoField<AttributeProto> attributes;
    for (const auto &[name, branch] :
         {std::pair<const char *, const char *>{"then_branch_ref", "left"},
          {"else_branch_ref", "right"}}) {
      auto &attribute = attributes.add();
      attribute.set_name(name);
      attribute.set_type(AttributeProto::STRING);
      attribute.set_s(branch);
    }
    source.MakeNode("If", {"condition"}, {"y"}, "", "", attributes);
    source.MakeOutput("y");
    GraphBuilder target(source.ToModel(), SchemaLookup());
    if (compatible) {
      ASSERT_NO_THROW(target.CopyDeviceTargetsFrom(source));
      EXPECT_EQ(target.LocalFunction("F").Subgraph("then").device(), Device::kCPU);
      EXPECT_EQ(target.LocalFunction("F").Subgraph("else").device(), Device::kGPU0);
    } else {
      EXPECT_THROW(target.CopyDeviceTargetsFrom(source), core::builder::BuilderError);
      EXPECT_EQ(target.LocalFunction("F").device(), Device::kUndefined);
      for (const auto &branch : target.LocalFunction("F").Subgraphs())
        EXPECT_EQ(branch->device(), Device::kUndefined);
    }
  }
}

TEST(GraphBuilder, ConstructorOpsets) {
  for (const std::string domain : {"", "ai.onnx"}) {
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{domain, 10}, {"com.example", 7}});
    EXPECT_EQ(builder.OpsetVersion(""), 10);
    EXPECT_EQ(builder.OpsetVersion("ai.onnx"), 10);
    EXPECT_EQ(builder.OpsetVersion("com.example"), 7);
    EXPECT_EQ(builder.Shapes().OpsetVersion(""), 10);
    EXPECT_EQ(builder.Shapes().OpsetVersion("com.example"), 7);
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    AttributeProto min;
    min.set_name("min");
    min.set_type(AttributeProto_AttributeType_FLOAT);
    min.set_f(0.0f);
    builder.MakeNode("Clip", {"x"}, {"y"}, "", "", std::vector<AttributeProto>{min});
    builder.MakeOutput("y");
    const ModelProto model = builder.ToModel();
    bool found_default = false;
    bool found_custom = false;
    for (const auto &opset : model.opset_import()) {
      if (opset.domain().empty() || opset.domain() == "ai.onnx") {
        EXPECT_EQ(opset.version(), 10);
        found_default = true;
      } else if (opset.domain() == "com.example") {
        EXPECT_EQ(opset.version(), 7);
        found_custom = true;
      }
    }
    EXPECT_TRUE(found_default);
    EXPECT_TRUE(found_custom);
  }
}

TEST(GraphBuilder, ConstructorDefaultOpset) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 17}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeNode("Relu", {"x"}, {"y"});
  EXPECT_EQ(builder.OpsetVersion(""), 17);
  builder.SetOpsetVersion("", 18);
  EXPECT_EQ(builder.OpsetVersion(""), 18);

  core::builder::GraphBuilder without_schema("g", {}, {{"", 18}});
  EXPECT_EQ(without_schema.OpsetVersion(""), 18);
}

TEST(GraphBuilder, MissingOpsetRejectsNodeBeforeMutation) {
  for (bool with_schema : {false, true}) {
    core::builder::GraphBuilder builder(
        "g", with_schema ? SchemaLookup() : core::builder::GraphBuilder::SchemaLookupFn{}, {});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
    EXPECT_THROW(builder.MakeNode("Relu", {"x"}, {"y"}), core::builder::BuilderError);
    EXPECT_TRUE(builder.Nodes().empty());
    EXPECT_TRUE(builder.OpsetVersions().empty());
    EXPECT_FALSE(builder.HasName("y"));
    builder.SetOpsetVersion("", 18);
    builder.MakeNode("Relu", {"x"}, {"y"});
    EXPECT_THROW(builder.MakeNode("Custom", {"y"}, {"z"}, "custom"), core::builder::BuilderError);
    EXPECT_FALSE(builder.HasName("z"));
    builder.SetOpsetVersion("custom", 1);
    builder.MakeNode("Custom", {"y"}, {"z"}, "custom");
    EXPECT_EQ(builder.Nodes().size(), 2u);
  }
}

TEST(GraphBuilder, RejectsNonpositiveOpsetsBeforeChangingImports) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  for (int version : {0, -1}) {
    EXPECT_THROW(builder.SetOpsetVersion("", version), core::builder::BuilderError);
    EXPECT_EQ(builder.OpsetVersion(""), 18);
    EXPECT_EQ(builder.Shapes().OpsetVersion(""), 18);
  }
}

TEST(GraphBuilder, LocalFunctionsInheritExplicitOpsetsWithoutInventingImports) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  auto &function = builder.MakeLocalFunction("F", "local");
  EXPECT_EQ(function.OpsetVersion(""), 18);
  EXPECT_EQ(builder.OpsetVersion("local"), core::shapes::kUnknownOpsetVersion);
  function.MakeInput("arg", core::symbolic::TensorType::kFloat, MakeShape({2}));
  function.MakeNode("Relu", {"arg"}, {"result"});
  function.MakeOutput("result");
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  EXPECT_THROW(builder.MakeNode("F", {"x"}, {"y"}, "local"), core::builder::BuilderError);
  EXPECT_FALSE(builder.HasName("y"));
  builder.SetOpsetVersion("local", 7);
  builder.MakeNode("F", {"x"}, {"y"}, "local");
}

TEST(GraphBuilder, NamesAreNeverReused) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  EXPECT_TRUE(builder.HasName("x"));
  EXPECT_THROW(builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3})),
               core::builder::BuilderError);
}

TEST(GraphBuilder, UniqueNameNeverCollides) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  const std::string a = builder.UniqueName("t");
  const std::string b = builder.UniqueName("t");
  EXPECT_NE(a, b);
  EXPECT_TRUE(builder.HasName(a));
  EXPECT_TRUE(builder.HasName(b));
}

TEST(GraphBuilder, MakeNodeUsesExplicitOpsetAndInfersShape) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const std::vector<std::string> outputs = builder.MakeNode("Add", {"x", "y"});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_TRUE(builder.HasName(outputs[0]));

  EXPECT_EQ(builder.OpsetVersion(""), 28);

  // The output shape was inferred incrementally.
  ASSERT_TRUE(builder.HasShape(outputs[0]));
  const core::symbolic::SymTensor &z = builder.GetShape(outputs[0]);
  EXPECT_EQ(z.Shape().Rank(), 2u);
}

TEST(GraphBuilder, SetOpsetVersionRebuildsMetadataImmediately) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"result"});
  builder.MakeOutput("result");

  builder.SetOpsetVersion("", 17);

  EXPECT_EQ(builder.OpsetVersion(""), 17);
  const GraphProto graph = builder.BuildGraph();
  core::compute::ComputeContext expected;
  const auto tags = expected.ComputeValueAndNodeTags(graph);
  expected.ComputeInPlaceReuseGraph(graph, builder.Shapes(), false, tags.first);
  EXPECT_EQ(builder.Compute().Reuse(), expected.Reuse());
  EXPECT_EQ(builder.Compute().Memory(), expected.Memory());
}

TEST(GraphBuilder, SetOpsetVersionUpdatesExistingNestedBuildersWithoutChangingFunctions) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  auto &body = builder.MakeSubgraph("body");
  body.MakeNode("Abs", {"x"}, {"body_result"});
  body.MakeOutput("body_result");
  auto &nested = body.MakeSubgraph("nested");
  nested.MakeNode("Identity", {"body_result"}, {"nested_result"});
  nested.MakeOutput("nested_result");
  auto &function = builder.MakeLocalFunction("F", "local");
  auto &function_body = function.MakeSubgraph("function_body");
  auto &nested_function = body.MakeLocalFunction("NestedF", "local");

  for (auto *scope : {&body, &nested}) {
    auto &shapes = const_cast<core::shapes::ShapesContext &>(scope->Shapes());
    shapes.set_events_enabled(true);
    shapes.ClearEvents();
  }
  builder.SetOpsetVersion("ai.onnx", 17);

  for (auto *scope : {&builder, &body, &nested}) {
    EXPECT_EQ(scope->OpsetVersion(""), 17);
    EXPECT_EQ(scope->Shapes().OpsetVersion("ai.onnx"), 17);
  }
  for (auto *scope : {&function, &function_body, &nested_function}) {
    EXPECT_EQ(scope->OpsetVersion(""), 18);
    EXPECT_EQ(scope->Shapes().OpsetVersion(""), 18);
  }
  for (auto *scope : {&body, &nested}) {
    EXPECT_EQ(std::count_if(scope->Shapes().Events().begin(), scope->Shapes().Events().end(),
                            [](const auto &event) {
                              return event.action == core::shapes::ShapeEventAction::kComputeNode;
                            }),
              1);
    EXPECT_EQ(scope->GetShape(scope->Outputs()[0].name().value()).Shape(), MakeShape({2, 3}));
    core::compute::ComputeContext expected;
    const GraphProto graph = scope->BuildGraph();
    const auto tags = expected.ComputeValueAndNodeTags(graph);
    expected.ComputeInPlaceReuseGraph(graph, scope->Shapes(), false, tags.first);
    EXPECT_EQ(scope->Compute().Reuse(), expected.Reuse());
    EXPECT_EQ(scope->Compute().Memory(), expected.Memory());
    const_cast<core::shapes::ShapesContext &>(scope->Shapes()).ClearEvents();
  }

  builder.SetOpsetVersion("", 17);
  EXPECT_TRUE(body.Shapes().Events().empty());
  EXPECT_TRUE(nested.Shapes().Events().empty());
  builder.SetOpsetVersion("custom", 2);
  for (auto *scope : {&builder, &body, &nested}) {
    EXPECT_EQ(scope->OpsetVersion("custom"), 2);
    EXPECT_EQ(scope->Shapes().OpsetVersion("custom"), 2);
    EXPECT_TRUE(scope->Shapes().Events().empty());
  }
  for (auto *scope : {&function, &function_body, &nested_function}) {
    EXPECT_EQ(scope->OpsetVersion("custom"), core::shapes::kUnknownOpsetVersion);
  }
  EXPECT_THROW(builder.SetOpsetVersion("", 0), core::builder::BuilderError);
  EXPECT_EQ(nested.OpsetVersion(""), 17);
  function.SetOpsetVersion("", 16);
  EXPECT_EQ(function_body.OpsetVersion(""), 16);
  EXPECT_EQ(function_body.Shapes().OpsetVersion(""), 16);
  EXPECT_EQ(builder.OpsetVersion(""), 17);
  EXPECT_EQ(nested_function.OpsetVersion(""), 18);

  nested.SetOpsetVersion("nested.new", 1);
  for (auto *scope : {&builder, &body, &nested})
    EXPECT_EQ(scope->OpsetVersion("nested.new"), 1);
  for (auto *scope : {&function, &function_body, &nested_function})
    EXPECT_EQ(scope->OpsetVersion("nested.new"), core::shapes::kUnknownOpsetVersion);
  function_body.SetOpsetVersion("nested.new", 2);
  EXPECT_EQ(function.OpsetVersion("nested.new"), 2);
  EXPECT_EQ(function_body.OpsetVersion("nested.new"), 2);
  EXPECT_EQ(builder.OpsetVersion("nested.new"), 1);
  EXPECT_EQ(nested_function.OpsetVersion("nested.new"), core::shapes::kUnknownOpsetVersion);
}

TEST(GraphBuilder, SetOpsetVersionRefreshesOnlyAffectedNestedConsumers) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}, {"local", 1}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("cond", core::symbolic::TensorType::kBool, MakeShape({}));
  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");

  auto &body = builder.MakeSubgraph("body");
  auto &nested = body.MakeSubgraph("nested");
  nested.MakeNode("F", {"x"}, {"nested_result"}, "local");
  nested.MakeOutput("nested_result");
  const auto make_if = [](core::builder::GraphBuilder &scope, const std::string &branch,
                          const std::string &output) {
    utils::RepeatedProtoField<AttributeProto> attributes;
    for (const char *name : {"then_branch_ref", "else_branch_ref"}) {
      auto &attribute = attributes.add();
      attribute.set_name(name);
      attribute.set_type(AttributeProto::STRING);
      attribute.set_s(branch);
    }
    scope.MakeNode("If", {"cond"}, {output}, "", "", attributes);
  };
  make_if(body, "nested", "body_result");
  body.MakeOutput("body_result");
  make_if(builder, "body", "result");
  make_if(builder, "body", "other_result");
  builder.MakeNode("Identity", {"result"}, {"dependent"});
  builder.MakeNode("Abs", {"x"}, {"independent"});
  builder.MakeOutput("dependent");

  for (auto *scope : {&builder, &body, &nested, &function}) {
    auto &shapes = const_cast<core::shapes::ShapesContext &>(scope->Shapes());
    shapes.set_events_enabled(true);
    shapes.ClearEvents();
  }
  builder.SetOpsetVersion("local", 2);

  const auto computations = [](const core::builder::GraphBuilder &scope) {
    std::vector<std::string> result;
    for (const auto &event : scope.Shapes().Events()) {
      if (event.action == core::shapes::ShapeEventAction::kComputeNode &&
          event.subgraph_node_index == -1) {
        result.push_back(event.op_type);
      }
    }
    return result;
  };
  EXPECT_EQ(computations(nested), (std::vector<std::string>{"F"}));
  EXPECT_EQ(computations(body), (std::vector<std::string>{"If"}));
  EXPECT_EQ(computations(builder), (std::vector<std::string>{"If", "If", "Identity"}));
  EXPECT_TRUE(computations(function).empty());
  EXPECT_EQ(function.OpsetVersion("local"), 1);
  EXPECT_EQ(builder.GetShape("dependent").Shape(), MakeShape({2, 3}));
  for (auto *scope : {&builder, &body, &nested}) {
    EXPECT_EQ(scope->OpsetVersion("local"), 2);
    EXPECT_EQ(scope->Shapes().OpsetVersion("local"), 2);
    const_cast<core::shapes::ShapesContext &>(scope->Shapes()).ClearEvents();
  }
  builder.SetOpsetVersion("unused", 1);
  for (auto *scope : {&builder, &body, &nested}) {
    EXPECT_TRUE(computations(*scope).empty());
    EXPECT_EQ(scope->OpsetVersion("unused"), 1);
  }
}

TEST(GraphBuilder, SetOpsetVersionInvalidatesChangedAndCapturedByteSizes) {
  for (bool symbolic : {false, true}) {
    core::builder::GraphBuilder builder("g", {}, {{"", 18}, {"versioned", 1}});
    const core::symbolic::SymDim extent =
        symbolic ? core::symbolic::SymDim("N") : core::symbolic::SymDim(4);
    const core::symbolic::SymShape fixed_shape{extent, core::symbolic::SymDim(1)};
    core::shapes::RegisterComputeShapeFn(
        "versioned", "Resize", [extent](core::shapes::ShapesContext &ctx, const NodeProto &node) {
          const int version = ctx.OpsetVersion("versioned");
          ctx.Set(node.output(0),
                  core::symbolic::SymTensor(nullptr,
                                            version == 3 ? core::symbolic::TensorType::kDouble
                                                         : core::symbolic::TensorType::kFloat,
                                            {extent, core::symbolic::SymDim(version)}));
        });
    core::shapes::RegisterComputeShapeFn(
        "versioned", "Fixed",
        [fixed_shape](core::shapes::ShapesContext &ctx, const NodeProto &node) {
          ctx.Set(node.output(0), core::symbolic::SymTensor(
                                      nullptr, core::symbolic::TensorType::kFloat, fixed_shape));
        });
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, fixed_shape);
    builder.MakeNode("Resize", {"x"}, {"changing"}, "versioned");
    auto &body = builder.MakeSubgraph("body");
    body.MakeNode("Identity", {"changing"}, {"captured"});
    body.MakeNode("Fixed", {"captured"}, {"body_result"}, "versioned");
    body.MakeOutput("body_result");
    builder.MakeNode("Fixed", {"changing"}, {"result"}, "versioned");
    builder.MakeOutput("result");
    ASSERT_EQ(builder.Compute().NodeReuse(1).size(), 1u);
    ASSERT_EQ(body.Compute().NodeReuse(1).size(), 1u);
    const auto previous_memory = builder.Compute().Memory();

    for (int version : {2, 3, 1}) {
      builder.SetOpsetVersion("versioned", version);
      EXPECT_EQ(body.GetShape("changing"), builder.GetShape("changing"));
      EXPECT_EQ(body.GetShape("captured").Shape(), builder.GetShape("changing").Shape());
      for (auto *scope : {&builder, &body}) {
        core::compute::ComputeContext expected;
        const GraphProto graph = scope->BuildGraph();
        const auto tags = expected.ComputeValueAndNodeTags(graph);
        expected.ComputeInPlaceReuseGraph(graph, scope->Shapes(), false, tags.first);
        EXPECT_EQ(scope->Compute().Reuse(), expected.Reuse());
        EXPECT_EQ(scope->Compute().Memory(), expected.Memory());
        if (symbolic && version != 1) {
          EXPECT_TRUE(scope->Compute().NodeReuse(1).empty());
        }
      }
      if (version == 2) {
        EXPECT_NE(builder.Compute().Memory(), previous_memory);
      }
    }
  }
}

TEST(GraphBuilder, SetOpsetVersionRefreshesMetadataSuffixOnce) {
  const auto peak_calls = std::make_shared<std::size_t>(0);
  core::shapes::RegisterComputePeakMemoryFn(
      "opset_batch_test", "Probe", core::symbolic::Device::kCPU,
      [peak_calls](core::symbolic::Device, const std::vector<core::symbolic::SymShape> &) {
        ++*peak_calls;
        return 0;
      });
  core::builder::GraphBuilder builder("g", {}, {{"opset_batch_test", 1}});
  core::shapes::RegisterComputeShapeFn(
      "opset_batch_test", "Probe", [](core::shapes::ShapesContext &ctx, const NodeProto &node) {
        ctx.Set(node.output(0), core::symbolic::SymTensor(ctx.Get(node.input(0))));
      });
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({4}));
  std::string previous = "x";
  for (int i = 0; i < 64; ++i) {
    previous = builder.MakeNode("Probe", {previous}, {}, "opset_batch_test")[0];
  }
  builder.MakeOutput(previous);
  ModelProto model;
  model.add_opset("opset_batch_test", 1);
  *model.mutable_graph() = builder.BuildGraph();
  for (const auto &node : model.graph().node()) {
    auto *value = model.mutable_graph()->add_value_info();
    *value = model.graph().input(0);
    value->set_name(node.output(0));
  }
  builder = core::builder::GraphBuilder(model, {});
  *peak_calls = 0;

  builder.SetOpsetVersion("opset_batch_test", 2);

  EXPECT_EQ(*peak_calls, builder.Nodes().size());
  core::compute::ComputeContext expected;
  const GraphProto graph = builder.BuildGraph();
  const auto tags = expected.ComputeValueAndNodeTags(graph);
  expected.ComputeInPlaceReuseGraph(graph, builder.Shapes(), false, tags.first);
  EXPECT_EQ(builder.Compute().Reuse(), expected.Reuse());
  EXPECT_EQ(builder.Compute().Memory(), expected.Memory());
  builder.SetOpsetVersion("opset_batch_test", 2);
  EXPECT_EQ(*peak_calls, builder.Nodes().size());
  expected.RefreshNodes(graph.node(), {{0, {"x"}}}, builder.Shapes(), {},
                        core::symbolic::Device::kCPU);
  EXPECT_EQ(*peak_calls, builder.Nodes().size());
}

TEST(GraphBuilder, SetOpsetVersionReplacesOnlyAffectedShapeConstraints) {
  core::shapes::RegisterComputeShapeFn(
      "opset_constraint_test", "Grow", [](core::shapes::ShapesContext &ctx, const NodeProto &node) {
        const int version = ctx.OpsetVersion("opset_constraint_test");
        ctx.Set(node.output(0),
                core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                          MakeShape({4, version})));
        ctx.AddConstraint("version_axis", std::to_string(version));
      });
  core::builder::GraphBuilder builder("g", SchemaLookup(),
                                      {{"", 18}, {"opset_constraint_test", 1}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({4, 1}));
  builder.MakeInput("stable", core::symbolic::TensorType::kFloat, MakeShape({5}));
  builder.MakeInput("cond", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeNode("NonZero", {"stable"}, {"unrelated"});
  builder.MakeNode("Grow", {"x"}, {"changing"}, "opset_constraint_test");
  builder.MakeNode("NonZero", {"changing"}, {"nonzero"});
  auto &body = builder.MakeSubgraph("body");
  body.MakeNode("Identity", {"changing"}, {"body_result"});
  body.MakeNode("NonZero", {"changing"}, {"body_nonzero"});
  body.MakeOutput("body_result");
  auto &otherwise = builder.MakeSubgraph("otherwise");
  otherwise.MakeNode("Identity", {"x"}, {"else_result"});
  otherwise.MakeOutput("else_result");
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto &[name, branch] :
       {std::pair{"then_branch_ref", "body"}, std::pair{"else_branch_ref", "otherwise"}}) {
    auto &attribute = attributes.add();
    attribute.set_name(name);
    attribute.set_type(AttributeProto::STRING);
    attribute.set_s(branch);
  }
  builder.MakeNode("If", {"cond"}, {"result"}, "", "", attributes);
  builder.MakeOutput("result");
  ASSERT_TRUE(builder.Shapes().HasLessEqualConstraint("NonZero_nonzero_nnz", "4"));

  for (int version : {2, 3, 1}) {
    builder.SetOpsetVersion("opset_constraint_test", version);
    core::shapes::ShapesContext fresh;
    fresh.SetOpsetVersion("", 18);
    fresh.SetOpsetVersion("opset_constraint_test", version);
    fresh.ComputeShapeGraph(builder.BuildGraph());
    EXPECT_EQ(builder.Shapes().Constraints(), fresh.Constraints());
    EXPECT_EQ(builder.Shapes().LessEqualConstraints(), fresh.LessEqualConstraints());
    EXPECT_TRUE(builder.Shapes().HasLessEqualConstraint("NonZero_unrelated_nnz", "5"));
    EXPECT_TRUE(builder.Shapes().HasLessEqualConstraint("NonZero_nonzero_nnz",
                                                        std::to_string(4 * version)));
    EXPECT_TRUE(body.Shapes().HasLessEqualConstraint("NonZero_body_nonzero_nnz",
                                                     std::to_string(4 * version)));
    EXPECT_EQ(body.Shapes().LessEqualConstraintsSize(), 1u);
    if (version == 1) {
      EXPECT_FALSE(builder.Shapes().HasLessEqualConstraint("If_result_d1", "3"));
    } else {
      EXPECT_FALSE(builder.Shapes().HasLessEqualConstraint("NonZero_nonzero_nnz", "4"));
      EXPECT_TRUE(builder.Shapes().HasLessEqualConstraint("If_result_d1", std::to_string(version)));
    }
  }
}

TEST(GraphBuilder, MakeNodeMaintainsTagsAndReuseIncrementally) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const std::vector<std::string> shape_out = builder.MakeNode("Shape", {"x"});
  ASSERT_EQ(shape_out.size(), 1u);

  // The value / node tags ("shape_tag") are kept up to date as nodes are added:
  // the output of Shape carries the "shape" tag right after MakeNode.
  const auto &value_tags = builder.Compute().ValueTags();
  auto it = value_tags.find(shape_out[0]);
  ASSERT_NE(it, value_tags.end());
  EXPECT_EQ(it->second, "shape");

  // In-place reuse is also computed incrementally: one entry per node so far.
  EXPECT_EQ(builder.Compute().Size(), builder.Nodes().size());

  const std::vector<std::string> abs_out = builder.MakeNode("Abs", {"x"});
  ASSERT_EQ(abs_out.size(), 1u);
  EXPECT_EQ(builder.Compute().Size(), builder.Nodes().size());
}

TEST(GraphBuilder, LaterConsumerRevokesEarlierIncrementalReuse) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"intermediate"});
  builder.MakeNode("Abs", {"intermediate"}, {"first"});
  ASSERT_EQ(builder.Compute().NodeReuse(1).size(), 1u);

  builder.MakeNode("Abs", {"intermediate"}, {"second"});
  EXPECT_TRUE(builder.Compute().NodeReuse(1).empty());
  builder.MakeOutput("first");
  builder.MakeOutput("second");
  core::compute::ComputeContext expected;
  expected.ComputeInPlaceReuseGraph(builder.BuildGraph(), builder.Shapes());
  EXPECT_EQ(builder.Compute().Reuse().size(), expected.Reuse().size());
  EXPECT_EQ(builder.Compute().NodeReuse(1).size(), expected.NodeReuse(1).size());
  const auto graph = builder.ToGraph();
  EXPECT_EQ(builder.Compute().Reuse().size(), graph.node().size());
  EXPECT_TRUE(builder.Compute().NodeReuse(1).empty());
}

TEST(GraphBuilder, UnusedShapeOutputReleaseMovesToConsumerAndRespectsGraphOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"data"});
  builder.MakeNode("Shape", {"data"}, {"dimensions"});
  EXPECT_EQ(builder.Compute().NodeReleaseAfterShapeTagged(1),
            (std::vector<std::string>{"dimensions"}));

  builder.MakeNode("Reshape", {"data", "dimensions"}, {"result"});
  EXPECT_TRUE(builder.Compute().NodeReleaseAfterShapeTagged(1).empty());
  EXPECT_EQ(builder.Compute().NodeReleaseAfterShapeTagged(2),
            (std::vector<std::string>{"dimensions"}));
  core::compute::ComputeContext before_output;
  before_output.ComputeInPlaceReuseGraph(builder.BuildGraph(), builder.Shapes(), false,
                                         builder.Compute().ValueTags());
  EXPECT_EQ(builder.Compute().Memory(), before_output.Memory());
  builder.MakeOutput("dimensions");
  EXPECT_TRUE(builder.Compute().NodeReleaseAfterShapeTagged(1).empty());
  EXPECT_TRUE(builder.Compute().NodeReleaseAfterShapeTagged(2).empty());

  core::compute::ComputeContext expected;
  const GraphProto graph = builder.BuildGraph();
  const auto tags = expected.ComputeValueAndNodeTags(graph);
  expected.ComputeInPlaceReuseGraph(graph, builder.Shapes(), false, tags.first);
  EXPECT_EQ(builder.Compute().ReleaseAfterShapeTagged(), expected.ReleaseAfterShapeTagged());
  EXPECT_EQ(builder.Compute().Memory(), expected.Memory());
}

TEST(GraphBuilder, RemovingLastConsumerMovesReleaseBackToProducer) {
  for (const bool reindex : {false, true}) {
    SCOPED_TRACE(reindex);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({1}));
    builder.MakeInput("D", core::symbolic::TensorType::kInt64, MakeShape({1}));
    builder.MakeNode("Shape", {"X"}, {"S"});
    builder.MakeNode("Identity", {"S"}, {"Y"});
    builder.MakeOutput("Y");
    auto &ctx = const_cast<core::compute::ComputeContext &>(builder.Compute());
    EXPECT_EQ(ctx.NodeReleaseAfterShapeTagged(1), (std::vector<std::string>{"S"}));

    GraphProto graph = builder.BuildGraph();
    *graph.mutable_node(1)->mutable_input(0) = "D";
    if (reindex) {
      ctx.UpdateNodes(graph.node(), {0, 1}, {{"X"}, {"D"}}, {}, core::symbolic::Device::kCPU);
      ctx.RefreshIncrementalMemory(graph.node(), ctx.Shapes(), core::symbolic::Device::kCPU);
    } else {
      ctx.RefreshNodes(graph.node(), {{1, {"D"}}}, ctx.Shapes(), {}, core::symbolic::Device::kCPU);
    }
    EXPECT_EQ(ctx.NodeReleaseAfterShapeTagged(0), (std::vector<std::string>{"S"}));
    EXPECT_TRUE(ctx.NodeReleaseAfterShapeTagged(1).empty());

    core::compute::ComputeContext expected;
    expected.ComputeInPlaceReuseGraph(graph, ctx.Shapes(), false, ctx.ValueTags());
    EXPECT_EQ(ctx.Memory(), expected.Memory());
    EXPECT_EQ(ctx.Reuse(), expected.Reuse());
  }
}

TEST(GraphBuilder, RepeatedInputsHaveOneReleaseAnnotation) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"intermediate"});
  builder.MakeNode("Add", {"intermediate", "intermediate"}, {"result"});
  builder.MakeOutput("result");
  const GraphProto graph = builder.ToGraph();
  const auto release = std::find_if(
      graph.node(1).metadata_props().begin(), graph.node(1).metadata_props().end(),
      [](const auto &entry) { return entry.key() == core::compute::kReleaseAfterMetadataKey; });
  ASSERT_NE(release, graph.node(1).metadata_props().end());
  EXPECT_EQ(release->value(), "intermediate");
}

TEST(GraphBuilder, OutputDeclaredAfterNodesRevokesIncrementalReuse) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"intermediate"});
  builder.MakeNode("Abs", {"intermediate"}, {"result"});
  ASSERT_EQ(builder.Compute().NodeReuse(1).size(), 1u);

  builder.MakeOutput("intermediate");
  EXPECT_TRUE(builder.Compute().NodeReuse(1).empty());
  builder.MakeOutput("result");
  core::compute::ComputeContext expected;
  expected.ComputeInPlaceReuseGraph(builder.BuildGraph(), builder.Shapes());
  EXPECT_EQ(builder.Compute().NodeReuse(1).size(), expected.NodeReuse(1).size());
}

TEST(GraphBuilder, DeclaredOutputShapeUpdatesMetadataImmediately) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("custom", 1);
  builder.MakeNode("Unknown", {}, {"intermediate"}, "custom");
  builder.MakeNode("Abs", {"intermediate"}, {"result"});
  EXPECT_FALSE(builder.HasShape("intermediate"));
  EXPECT_FALSE(builder.HasShape("result"));

  ValueInfoProto output;
  output.set_name("intermediate");
  auto *tensor = output.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::DataType::FLOAT);
  tensor->mutable_shape()->add_dim()->set_dim_value(2);
  tensor->mutable_shape()->add_dim()->set_dim_value(3);
  builder.MakeOutput(output);

  ASSERT_TRUE(builder.HasShape("intermediate"));
  EXPECT_EQ(builder.GetShape("intermediate").Shape(), MakeShape({2, 3}));
  ASSERT_TRUE(builder.HasShape("result"));
  EXPECT_EQ(builder.GetShape("result").Shape(), MakeShape({2, 3}));
  ASSERT_EQ(builder.Compute().Memory().size(), 2u);
}

TEST(GraphBuilder, ConflictingTypedOutputAfterInferenceIsRejectedBeforeMutation) {
  for (int conflict = 0; conflict < 6; ++conflict) {
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    builder.MakeNode("Abs", {"x"}, {"z"});

    ValueInfoProto output;
    output.set_name("z");
    auto *tensor = output.mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(conflict < 2 ? TensorProto::INT64 : TensorProto::FLOAT);
    if (conflict == 5) {
      tensor->mutable_shape();
    } else if (conflict != 1) {
      tensor->mutable_shape()->add_dim()->set_dim_value(conflict == 2 ? 4 : 2);
      if (conflict != 3) {
        tensor->mutable_shape()->add_dim()->set_dim_value(conflict == 4 ? 5 : 3);
      }
    }
    EXPECT_THROW(builder.MakeOutput(output), core::builder::BuilderError) << conflict;
    EXPECT_TRUE(builder.Outputs().empty());
    EXPECT_TRUE(builder.BuildGraph().output().empty());
    EXPECT_EQ(builder.GetShape("z").Dtype(), core::symbolic::TensorType::kFloat);
    EXPECT_EQ(builder.GetShape("z").Shape(), MakeShape({2, 3}));

    builder.MakeOutput("z", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    const auto expected = builder.Outputs()[0];
    EXPECT_THROW(builder.MakeOutput(output), core::builder::BuilderError) << conflict;
    ASSERT_EQ(builder.Outputs().size(), 1u);
    EXPECT_TRUE(builder.Outputs()[0].Equals(expected));
    EXPECT_TRUE(builder.BuildGraph().output(0).type().Equals(expected.type()));
    EXPECT_TRUE(builder.ToModel().graph().output(0).type().Equals(expected.type()));
  }
}

TEST(GraphBuilder, CompatibleTypedOutputAfterInferenceKeepsPublicViewsCoherent) {
  for (int inference = 0; inference < 3; ++inference) {
    for (int declaration = 0; declaration < 5; ++declaration) {
      core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
      core::symbolic::SymShape shape = MakeShape({2, 3});
      if (inference != 0) {
        shape[0] = core::symbolic::SymDim(inference == 1 ? "batch" : "");
      }
      builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
      builder.MakeNode("Abs", {"x"}, {"z"});

      ValueInfoProto output;
      output.set_name("z");
      output.set_doc_string("declared output");
      auto *tensor = output.mutable_type()->mutable_tensor_type();
      tensor->set_elem_type(declaration == 4 ? TensorProto::UNDEFINED : TensorProto::FLOAT);
      if (declaration != 3) {
        auto *dim = tensor->mutable_shape()->add_dim();
        if (declaration == 1) {
          dim->set_dim_param("declared_batch");
        } else if (declaration != 2) {
          dim->set_dim_value(2);
        }
        tensor->mutable_shape()->add_dim()->set_dim_value(3);
      }
      builder.MakeOutput(output);

      EXPECT_EQ(builder.GetShape("z").Dtype(), core::symbolic::TensorType::kFloat);
      EXPECT_EQ(builder.GetShape("z").Shape(), shape);
      ValueInfoProto inferred;
      ASSERT_TRUE(core::symbolic::SymTensorToValueInfo(builder.GetShape("z"), inferred));
      ASSERT_EQ(builder.Outputs().size(), 1u);
      EXPECT_EQ(builder.Outputs()[0].doc_string().value(), "declared output");
      EXPECT_TRUE(builder.Outputs()[0].type().Equals(inferred.type()));
      EXPECT_TRUE(builder.BuildGraph().output(0).type().Equals(inferred.type()));
      EXPECT_TRUE(builder.ToModel().graph().output(0).type().Equals(inferred.type()));
    }
  }
}

TEST(GraphBuilder, TypedOutputRefinesUndefinedDtypeAndOnlyAffectedConsumers) {
  for (bool declare_shape : {false, true}) {
    for (auto dtype : {TensorProto::FLOAT, TensorProto::DOUBLE}) {
      core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
      ValueInfoProto input;
      input.set_name("x");
      auto *input_shape = input.mutable_type()->mutable_tensor_type()->mutable_shape();
      input_shape->add_dim()->set_dim_value(2);
      input_shape->add_dim()->set_dim_value(3);
      builder.MakeInput(input);
      builder.MakeInput("other", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
      builder.MakeNode("Abs", {"x"}, {"z"});
      builder.MakeNode("Identity", {"z"}, {"result"});
      builder.MakeNode("Abs", {"other"}, {"independent"});
      ASSERT_EQ(builder.GetShape("z").Dtype(), core::symbolic::TensorType::kUndefined);
      ValueInfoProto result = input;
      result.set_name("result");
      builder.MakeOutput(result);

      ValueInfoProto output;
      output.set_name("z");
      output.set_doc_string("declared output");
      output.mutable_type()->mutable_tensor_type()->set_elem_type(dtype);
      if (declare_shape) {
        *output.mutable_type()->mutable_tensor_type()->mutable_shape() = *input_shape;
      }
      auto &shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
      shapes.set_events_enabled(true);
      shapes.ClearEvents();
      ValueInfoProto incompatible = output;
      auto *incompatible_shape =
          incompatible.mutable_type()->mutable_tensor_type()->mutable_shape();
      *incompatible_shape = *input_shape;
      incompatible_shape->mutable_dim(0)->set_dim_value(4);
      const auto memory = builder.Compute().Memory();
      EXPECT_THROW(builder.MakeOutput(incompatible), core::builder::BuilderError);
      incompatible = output;
      incompatible.mutable_type()->mutable_tensor_type()->set_elem_type(
          static_cast<TensorProto::DataType>(999));
      EXPECT_THROW(builder.MakeOutput(incompatible), core::builder::BuilderError);
      EXPECT_TRUE(shapes.Events().empty());
      EXPECT_EQ(builder.Outputs().size(), 1u);
      EXPECT_EQ(builder.GetShape("z").Dtype(), core::symbolic::TensorType::kUndefined);
      EXPECT_EQ(builder.Compute().Memory(), memory);
      builder.MakeOutput(output);

      std::vector<std::string> computed;
      for (const auto &event : shapes.Events()) {
        if (event.action == core::shapes::ShapeEventAction::kComputeNode) {
          computed.push_back(event.op_type);
        }
      }
      EXPECT_EQ(computed, (std::vector<std::string>{"Identity"}));
      EXPECT_EQ(builder.GetShape("x").Dtype(), core::symbolic::TensorType::kUndefined);
      for (int refresh : {0, 1, 2}) {
        SCOPED_TRACE(refresh);
        if (refresh == 1) {
          builder.SetOpsetVersion("", 27);
        } else if (refresh == 2) {
          builder.RemoveUnusedNodes();
          EXPECT_EQ(builder.RemoveUnusedNodes(), 0u);
        }
        const GraphProto graph = builder.BuildGraph();
        const ModelProto model = builder.ToModel();
        ASSERT_EQ(builder.Outputs().size(), 2u);
        EXPECT_EQ(builder.Outputs()[1].doc_string().value(), "declared output");
        for (std::size_t i = 0; i < builder.Outputs().size(); ++i) {
          const auto &name = builder.Outputs()[i].name().value();
          EXPECT_EQ(builder.GetShape(name).Dtype(), core::symbolic::DataTypeToTensorType(dtype));
          EXPECT_EQ(builder.GetShape(name).Shape(), MakeShape({2, 3}));
          ValueInfoProto inferred;
          ASSERT_TRUE(core::symbolic::SymTensorToValueInfo(builder.GetShape(name), inferred));
          EXPECT_TRUE(builder.Outputs()[i].type().Equals(inferred.type()));
          EXPECT_TRUE(graph.output(i).type().Equals(inferred.type()));
          EXPECT_TRUE(model.graph().output(i).type().Equals(inferred.type()));
        }
        core::compute::ComputeContext expected;
        const auto tags = expected.ComputeValueAndNodeTags(graph);
        expected.ComputeInPlaceReuseGraph(graph, builder.Shapes(), false, tags.first);
        EXPECT_EQ(builder.Compute().Reuse(), expected.Reuse());
        EXPECT_EQ(builder.Compute().Memory(), expected.Memory());
      }
    }
  }
}

TEST(GraphBuilder, InputOutputDtypeRefinementPrecedesRebuiltConsumers) {
  for (auto dtype : {core::symbolic::TensorType::kFloat, core::symbolic::TensorType::kDouble}) {
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    ValueInfoProto input;
    input.set_name("x");
    auto *shape = input.mutable_type()->mutable_tensor_type()->mutable_shape();
    shape->add_dim()->set_dim_value(2);
    shape->add_dim()->set_dim_value(3);
    builder.MakeInput(input);
    builder.MakeNode("Abs", {"x"}, {"z"});
    builder.MakeOutput("z");
    builder.MakeOutput("x", dtype, MakeShape({2, 3}));

    for (bool rebuild : {false, true}) {
      if (rebuild) {
        EXPECT_EQ(builder.RemoveUnusedNodes(), 0u);
      }
      EXPECT_FALSE(builder.Outputs()[0].has_type());
      EXPECT_EQ(builder.GetShape("x").Dtype(), dtype);
      EXPECT_EQ(builder.GetShape("z").Dtype(), dtype);
      EXPECT_EQ(builder.GetShape("z").Shape(), MakeShape({2, 3}));
      const GraphProto graph = builder.BuildGraph();
      core::compute::ComputeContext expected;
      const auto tags = expected.ComputeValueAndNodeTags(graph);
      core::shapes::ShapesContext expected_shapes;
      expected_shapes.Set("x", core::symbolic::SymTensor(nullptr, dtype, MakeShape({2, 3})));
      expected_shapes.Set("z", core::symbolic::SymTensor(nullptr, dtype, MakeShape({2, 3})));
      expected.ComputeInPlaceReuseGraph(graph, expected_shapes, false, tags.first);
      expected.ComputePeakMemory(graph, builder.device());
      EXPECT_EQ(builder.Compute().Reuse(), expected.Reuse());
      EXPECT_EQ(builder.Compute().Memory(), expected.Memory());
      EXPECT_EQ(builder.Compute().PeakMemory(), expected.PeakMemory());
    }
  }
}

TEST(GraphBuilder, ImportsSequenceAtOutputWithUnknownElementRank) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset("", 18);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("sequence_at");
  auto *sequence = graph->add_input();
  sequence->set_name("s");
  sequence->mutable_type()
      ->mutable_sequence_type()
      ->mutable_elem_type()
      ->mutable_tensor_type()
      ->set_elem_type(TensorProto::FLOAT);
  auto *position = graph->add_input();
  position->set_name("p");
  auto *position_type = position->mutable_type()->mutable_tensor_type();
  position_type->set_elem_type(TensorProto::INT64);
  position_type->mutable_shape();
  graph->add_node(MakeNode("SequenceAt", {"s", "p"}, {"z"}));
  graph->add_node(MakeNode("Identity", {"z"}, {"result"}));
  for (const char *name : {"z", "result"}) {
    auto *output = graph->add_output();
    output->set_name(name);
    auto *tensor = output->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(TensorProto::FLOAT);
    tensor->mutable_shape()->add_dim()->set_dim_value(2);
    tensor->mutable_shape()->add_dim()->set_dim_value(3);
  }

  core::builder::GraphBuilder builder(model, SchemaLookup());
  for (bool refresh : {false, true}) {
    if (refresh) {
      builder.SetOpsetVersion("", 17);
      EXPECT_EQ(builder.RemoveUnusedNodes(), 0u);
    }
    for (std::size_t i = 0; i < builder.Outputs().size(); ++i) {
      const auto &name = builder.Outputs()[i].name().value();
      EXPECT_EQ(builder.GetShape(name).Dtype(), core::symbolic::TensorType::kFloat);
      EXPECT_EQ(builder.GetShape(name).Shape(), MakeShape({2, 3}));
      EXPECT_TRUE(builder.Outputs()[i].type().Equals(graph->output(i).type()));
      EXPECT_TRUE(builder.BuildGraph().output(i).type().Equals(graph->output(i).type()));
      EXPECT_TRUE(builder.ToModel().graph().output(i).type().Equals(graph->output(i).type()));
    }
  }
}

TEST(GraphBuilder, SequenceAtUnknownRankStillValidatesDtypeAndKnownScalars) {
  core::builder::GraphBuilder builder("sequence_at", SchemaLookup(), {{"", 18}});
  ValueInfoProto sequence;
  sequence.set_name("s");
  sequence.mutable_type()
      ->mutable_sequence_type()
      ->mutable_elem_type()
      ->mutable_tensor_type()
      ->set_elem_type(TensorProto::FLOAT);
  builder.MakeInput(sequence);
  builder.MakeInput("p", core::symbolic::TensorType::kInt64, MakeShape({}));
  builder.MakeNode("SequenceAt", {"s", "p"}, {"z"});
  EXPECT_THROW(builder.MakeOutput("z", core::symbolic::TensorType::kDouble, MakeShape({2, 3})),
               core::builder::BuilderError);
  EXPECT_TRUE(builder.Outputs().empty());
  builder.MakeInput("scalar", core::symbolic::TensorType::kFloat, MakeShape({}));
  builder.MakeNode("SequenceConstruct", {"scalar"}, {"scalar_sequence"});
  builder.MakeNode("SequenceAt", {"scalar_sequence", "p"}, {"scalar_result"});
  EXPECT_THROW(
      builder.MakeOutput("scalar_result", core::symbolic::TensorType::kFloat, MakeShape({2, 3})),
      core::builder::BuilderError);
  EXPECT_TRUE(builder.Outputs().empty());

  sequence.set_name("scalar_input_sequence");
  sequence.mutable_type()
      ->mutable_sequence_type()
      ->mutable_elem_type()
      ->mutable_tensor_type()
      ->mutable_shape();
  builder.MakeInput(sequence);
  builder.MakeNode("SequenceAt", {"scalar_input_sequence", "p"}, {"input_scalar_result"});
  EXPECT_EQ(builder.GetShape("input_scalar_result").Shape().Rank(), 0u);
  EXPECT_THROW(builder.MakeOutput("input_scalar_result", core::symbolic::TensorType::kFloat,
                                  MakeShape({2, 3})),
               core::builder::BuilderError);
  EXPECT_TRUE(builder.Outputs().empty());
  builder.MakeOutput("input_scalar_result", core::symbolic::TensorType::kFloat, MakeShape({}));
  builder.RemoveUnusedNodes();
  EXPECT_EQ(builder.GetShape("input_scalar_result").Shape().Rank(), 0u);
  core::builder::GraphBuilder imported(builder.ToModel(), SchemaLookup());
  EXPECT_EQ(imported.GetShape("input_scalar_result").Shape().Rank(), 0u);
}

TEST(GraphBuilder, SequenceAtAnonymousOutputDimensionsSurviveImportAndRebuilding) {
  for (bool import_output : {false, true}) {
    ModelProto model;
    model.set_ir_version(10);
    model.add_opset("", 18);
    auto *graph = model.mutable_graph();
    graph->set_name("anonymous_output");
    auto *sequence = graph->add_input();
    sequence->set_name("s");
    sequence->mutable_type()
        ->mutable_sequence_type()
        ->mutable_elem_type()
        ->mutable_tensor_type()
        ->set_elem_type(TensorProto::FLOAT);
    auto *position = graph->add_input();
    position->set_name("p");
    auto *position_type = position->mutable_type()->mutable_tensor_type();
    position_type->set_elem_type(TensorProto::INT64);
    position_type->mutable_shape();
    graph->add_node(MakeNode("SequenceAt", {"s", "p"}, {"z"}));
    graph->add_node(MakeNode("Identity", {"z"}, {"result"}));
    ValueInfoProto output;
    output.set_name("z");
    auto *tensor = output.mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(TensorProto::FLOAT);
    tensor->mutable_shape()->add_dim();
    tensor->mutable_shape()->add_dim()->set_dim_value(3);
    if (import_output) {
      *graph->add_output() = output;
    }
    core::builder::GraphBuilder builder(model, SchemaLookup());
    if (!import_output) {
      builder.MakeOutput(output);
    }
    builder.MakeOutput("result");
    for (bool rebuild : {false, true}) {
      if (rebuild) {
        EXPECT_EQ(builder.RemoveUnusedNodes(), 0u);
      }
      for (const char *name : {"z", "result"}) {
        const auto &actual = builder.GetShape(name);
        EXPECT_EQ(actual.Dtype(), core::symbolic::TensorType::kFloat);
        ASSERT_EQ(actual.Shape().Rank(), 2u);
        EXPECT_EQ(actual.Shape()[0], core::symbolic::SymDim(""));
        EXPECT_EQ(actual.Shape()[1], core::symbolic::SymDim(3));
      }
      const auto exported = builder.ToModel();
      const auto &dimension = exported.graph().output(0).type().tensor_type().shape().dim(0);
      EXPECT_FALSE(dimension.has_dim_value());
      EXPECT_TRUE(dimension.has_dim_param());
      EXPECT_TRUE(dimension.dim_param().empty());
    }
  }
}

TEST(GraphBuilder, UnknownRankTensorConsumersKeepDtypeWithoutAnIntermediateDeclaration) {
  for (const std::string consumer : {"Identity", "Abs"}) {
    SCOPED_TRACE(consumer);
    core::builder::GraphBuilder builder("unknown_rank_consumer", SchemaLookup(), {{"", 18}});
    ValueInfoProto sequence;
    sequence.set_name("s");
    sequence.mutable_type()
        ->mutable_sequence_type()
        ->mutable_elem_type()
        ->mutable_tensor_type()
        ->set_elem_type(TensorProto::FLOAT);
    builder.MakeInput(sequence);
    builder.MakeInput("p", core::symbolic::TensorType::kInt64, MakeShape({}));
    builder.MakeNode("SequenceAt", {"s", "p"}, {"z"});
    builder.MakeNode(consumer, {"z"}, {"middle"});
    builder.MakeNode("Identity", {"middle"}, {"result"});
    const auto check = [&](const core::builder::GraphBuilder &actual) {
      for (const char *name : {"z", "middle", "result"}) {
        EXPECT_FALSE(actual.HasShape(name));
        ASSERT_TRUE(actual.Compute().Shapes().HasType(name));
        const auto &type = actual.Compute().Shapes().GetType(name);
        ASSERT_TRUE(type.has_tensor_type());
        EXPECT_EQ(type.tensor_type().elem_type(), TensorProto::FLOAT);
        EXPECT_FALSE(type.tensor_type().has_shape());
      }
    };
    const auto reject_conflict = [&] {
      const auto count = builder.Outputs().size();
      EXPECT_THROW(
          builder.MakeOutput("result", core::symbolic::TensorType::kDouble, MakeShape({2, 3})),
          core::builder::BuilderError);
      EXPECT_EQ(builder.Outputs().size(), count);
      check(builder);
    };
    check(builder);
    reject_conflict();
    builder.MakeOutput("result");
    builder.SetOpsetVersion("", 17);
    reject_conflict();
    EXPECT_EQ(builder.RemoveUnusedNodes(), 0u);
    reject_conflict();
    ModelProto exported = builder.ToModel();
    EXPECT_EQ(exported.graph().output(0).type().tensor_type().elem_type(), TensorProto::FLOAT);
    EXPECT_FALSE(exported.graph().output(0).type().tensor_type().has_shape());
    core::builder::GraphBuilder imported(exported, SchemaLookup());
    check(imported);
  }
}

TEST(GraphBuilder, TypedOutputRebuildDefersPersistentBindingValidationDuringImport) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset("", 23);
  model.add_opset("custom", 1);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("g");

  ValueInfoProto *past = graph->add_input();
  past->set_name("past");
  auto *past_tensor = past->mutable_type()->mutable_tensor_type();
  past_tensor->set_elem_type(TensorProto::DataType::FLOAT);
  past_tensor->mutable_shape()->add_dim()->set_dim_value(2);
  past_tensor->mutable_shape()->add_dim()->set_dim_value(3);

  graph->add_node(MakeNode("Unknown", {}, {"intermediate"}, "custom"));
  graph->add_node(MakeNode("Identity", {"past"}, {"present"}));

  ValueInfoProto *intermediate = graph->add_output();
  intermediate->set_name("intermediate");
  auto *intermediate_tensor = intermediate->mutable_type()->mutable_tensor_type();
  intermediate_tensor->set_elem_type(TensorProto::DataType::FLOAT);
  intermediate_tensor->mutable_shape()->add_dim()->set_dim_value(2);
  intermediate_tensor->mutable_shape()->add_dim()->set_dim_value(3);
  ValueInfoProto *present = graph->add_output();
  present->set_name("present");
  *present->mutable_type() = past->type();

  auto *binding = graph->add_persistent_bindings();
  binding->set_input_name("past");
  binding->set_output_name("present");

  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_EQ(builder.BuildGraph().output().size(), 2u);
}

TEST(GraphBuilder, DeclaredOutputTagPropagatesToEarlierNodes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"intermediate"});
  builder.MakeNode("Abs", {"intermediate"}, {"result"});

  ValueInfoProto output;
  output.set_name("intermediate");
  auto *tag = output.add_metadata_props();
  tag->set_key(core::compute::kValueTagMetadataKey);
  tag->set_value("shape");
  builder.MakeOutput(output);
  EXPECT_EQ(builder.Compute().ValueTags().at("intermediate"), "shape");
  EXPECT_EQ(builder.Compute().NodeTag(1), "shape");
  EXPECT_EQ(builder.Compute().ValueTags().at("result"), "shape");
  core::compute::ComputeContext expected;
  const auto whole_graph = expected.ComputeValueAndNodeTags(builder.BuildGraph());
  EXPECT_EQ(builder.Compute().ValueTags(), whole_graph.first);
  EXPECT_EQ(builder.Compute().NodeTags(), whole_graph.second);
}

TEST(GraphBuilder, LateTagRefreshesReleaseWithoutDownstreamConsumers) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"intermediate"});
  builder.MakeNode("Abs", {"intermediate"}, {"result"});
  EXPECT_TRUE(builder.Compute().NodeReleaseAfterShapeTagged(1).empty());

  auto &incremental = const_cast<core::compute::ComputeContext &>(builder.Compute());
  incremental.SeedValueTag("intermediate", "shape", builder.Nodes());
  EXPECT_EQ(incremental.NodeReleaseAfterShapeTagged(1),
            (std::vector<std::string>{"intermediate", "result"}));
}

TEST(GraphBuilder, RepeatedGraphExportPreservesMetadataAfterAppending) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"intermediate"});
  builder.MakeNode("Abs", {"intermediate"}, {"first"});
  builder.MakeOutput("first");

  const GraphProto initial = builder.ToGraph();
  EXPECT_EQ(builder.ToGraph().SerializeAsString(), initial.SerializeAsString());

  builder.MakeNode("Abs", {"intermediate"}, {"second"});
  builder.MakeOutput("second");
  core::compute::ComputeContext expected;
  expected.ComputeInPlaceReuseGraph(builder.BuildGraph(), builder.Shapes());
  ASSERT_EQ(builder.Compute().Reuse().size(), expected.Reuse().size());
  EXPECT_EQ(builder.Compute().NodeReuse(1).size(), expected.NodeReuse(1).size());
  const GraphProto extended = builder.ToGraph();
  EXPECT_EQ(builder.ToGraph().SerializeAsString(), extended.SerializeAsString());
}

TEST(GraphBuilder, MaintainsMemoryProfilesIncrementally) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"result"});
  builder.MakeOutput("result");

  const GraphProto graph = builder.BuildGraph();
  core::compute::ComputeContext expected;
  const auto tags = expected.ComputeValueAndNodeTags(graph);
  expected.ComputeInPlaceReuseGraph(graph, builder.Shapes(), false, tags.first);
  EXPECT_EQ(builder.Compute().Memory(), expected.Memory());
}

TEST(GraphBuilder, IncrementalMetadataRejectsGraphInputOverwrite) {
  core::compute::ComputeContext compute;
  EXPECT_THROW(compute.SeedReuseInput("x", true, false, true), std::invalid_argument);
}

TEST(GraphBuilder, RefreshesOnlyAffectedIncrementalMemoryRange) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  const auto expect_whole_graph_metadata = [&]() {
    const GraphProto graph = builder.BuildGraph();
    core::compute::ComputeContext expected;
    const auto tags = expected.ComputeValueAndNodeTags(graph);
    expected.ComputeInPlaceReuseGraph(graph, builder.Shapes(), false, tags.first);
    EXPECT_EQ(builder.Compute().Reuse(), expected.Reuse());
    EXPECT_EQ(builder.Compute().Memory(), expected.Memory());
  };

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"intermediate"});
  expect_whole_graph_metadata();
  builder.MakeNode("Abs", {"intermediate"}, {"first"});
  expect_whole_graph_metadata();
  builder.MakeNode("Abs", {"intermediate"}, {"second"});
  expect_whole_graph_metadata();
  builder.MakeOutput("intermediate");
  expect_whole_graph_metadata();
  builder.MakeInitializer(
      MakeInitializer<float>("late_weight", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  expect_whole_graph_metadata();

  ValueInfoProto output;
  output.set_name("second");
  auto *tag = output.add_metadata_props();
  tag->set_key(core::compute::kValueTagMetadataKey);
  tag->set_value("shape");
  builder.MakeOutput(output);
  expect_whole_graph_metadata();
}

TEST(GraphBuilder, AdvancingKeptInitializerUpdatesOnlyItsLastUse) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInitializer(
      MakeInitializer<float>("weight", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  builder.MakeNode("Add", {"x", "weight"}, {"first"});
  builder.MakeNode("Abs", {"first"}, {"middle"});
  builder.MakeNode("Add", {"middle", "weight"}, {"result"});
  builder.MakeOutput("result");

  const GraphProto graph = builder.BuildGraph();
  core::compute::ComputeContext expected;
  const auto tags = expected.ComputeValueAndNodeTags(graph);
  expected.ComputeInPlaceReuseGraph(graph, builder.Shapes(), false, tags.first);
  EXPECT_EQ(builder.Compute().Reuse(), expected.Reuse());
  EXPECT_EQ(builder.Compute().Memory(), expected.Memory());

  GraphProto annotated = graph;
  builder.Compute().WriteToGraph(annotated);
  const auto first_last_use = std::find_if(
      annotated.node()[0].metadata_props().begin(), annotated.node()[0].metadata_props().end(),
      [](const auto &entry) { return entry.key() == core::compute::kNotUsedAfterMetadataKey; });
  ASSERT_NE(first_last_use, annotated.node()[0].metadata_props().end());
  EXPECT_EQ(first_last_use->value(), "x");
  const auto final_last_use = std::find_if(
      annotated.node()[2].metadata_props().begin(), annotated.node()[2].metadata_props().end(),
      [](const auto &entry) { return entry.key() == core::compute::kNotUsedAfterMetadataKey; });
  ASSERT_NE(final_last_use, annotated.node()[2].metadata_props().end());
  EXPECT_EQ(final_last_use->value(), "weight");
}

TEST(GraphBuilder, ChildMutationUpdatesParentLifetimeMetadataImmediately) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("custom", 1);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Abs", {"x"}, {"intermediate"});
  builder.MakeNode("Abs", {"intermediate"}, {"result"});
  auto &child = builder.MakeSubgraph("body");
  child.MakeNode("Identity", {"x"}, {"child_x"});
  child.MakeOutput("child_x");
  utils::RepeatedProtoField<AttributeProto> attributes;
  auto &body = attributes.add();
  body.set_name("body_ref");
  body.set_type(AttributeProto::STRING);
  body.set_s("body");
  builder.MakeNode("Capture", {}, {"captured"}, "custom", "", attributes);
  builder.MakeOutput("result");
  builder.MakeOutput("captured");

  const GraphProto before = builder.ToGraph();
  child.MakeNode("Identity", {"intermediate"}, {"child_intermediate"});
  child.MakeOutput("child_intermediate");
  GraphProto after = builder.BuildGraph();
  builder.Compute().WriteToGraph(after);
  const auto released_before = std::find_if(
      before.node()[1].metadata_props().begin(), before.node()[1].metadata_props().end(),
      [](const auto &entry) { return entry.key() == core::compute::kReleaseAfterMetadataKey; });
  ASSERT_NE(released_before, before.node()[1].metadata_props().end());
  EXPECT_EQ(released_before->value(), "intermediate");
  const auto released_after = std::find_if(
      after.node()[1].metadata_props().begin(), after.node()[1].metadata_props().end(),
      [](const auto &entry) { return entry.key() == core::compute::kReleaseAfterMetadataKey; });
  EXPECT_EQ(released_after, after.node()[1].metadata_props().end());
}

TEST(GraphBuilder, ChildMutationUpdatesParentShapeMetadataImmediately) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("local", 1);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("F", {"x"}, {"result"}, "local");
  builder.MakeOutput("result");
  EXPECT_FALSE(builder.HasShape("result"));

  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");

  EXPECT_EQ(builder.GetShape("result").Shape(), MakeShape({2, 3}));
  GraphProto graph = builder.BuildGraph();
  builder.Compute().WriteToGraph(graph);
  ASSERT_EQ(graph.output().size(), 1u);
  ASSERT_TRUE(graph.output()[0].has_type());
  ASSERT_TRUE(graph.output()[0].type().has_tensor_type());
  EXPECT_EQ(graph.output()[0].type().tensor_type().shape().dim().size(), 2u);
}

TEST(GraphBuilder, ChildMutationRefreshesOnlyDependentParentNodes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("local", 1);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("unrelated", core::symbolic::TensorType::kFloat, MakeShape({4, 5}));
  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");
  builder.MakeNode("F", {"x"}, {"call"}, "local");
  builder.MakeNode("Identity", {"call"}, {"dependent"});
  builder.MakeNode("Abs", {"unrelated"}, {"independent"});

  auto &shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
  shapes.set_events_enabled(true);
  shapes.ClearEvents();
  function.MakeNode("Neg", {"a"}, {"dead"});

  std::vector<std::string> refreshed;
  for (const auto &event : shapes.Events()) {
    if (event.action == core::shapes::ShapeEventAction::kComputeNode) {
      refreshed.push_back(event.op_type);
    }
  }
  EXPECT_EQ(refreshed, (std::vector<std::string>{"F", "Identity"}));
  EXPECT_EQ(builder.GetShape("dependent").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(builder.GetShape("independent").Shape(), MakeShape({4, 5}));
}

TEST(GraphBuilder, LocalFunctionMutationRefreshesTransitiveCallers) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("local", 1);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  auto &inner = builder.MakeLocalFunction("Inner", "local");
  inner.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  inner.MakeNode("Identity", {"a"}, {"r"});
  inner.MakeOutput("r");
  auto &outer = builder.MakeLocalFunction("Outer", "local");
  outer.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  outer.MakeNode("Inner", {"a"}, {"r"}, "local");
  outer.MakeOutput("r");
  builder.MakeNode("Outer", {"x"}, {"result"}, "local");

  auto &root_shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
  auto &outer_shapes = const_cast<core::shapes::ShapesContext &>(outer.Shapes());
  root_shapes.set_events_enabled(true);
  outer_shapes.set_events_enabled(true);
  root_shapes.ClearEvents();
  outer_shapes.ClearEvents();
  inner.MakeNode("Neg", {"a"}, {"dead"});

  EXPECT_TRUE(
      std::any_of(root_shapes.Events().begin(), root_shapes.Events().end(), [](const auto &event) {
        return event.action == core::shapes::ShapeEventAction::kComputeNode &&
               event.op_type == "Outer";
      }));
  EXPECT_TRUE(std::any_of(outer_shapes.Events().begin(), outer_shapes.Events().end(),
                          [](const auto &event) {
                            return event.action == core::shapes::ShapeEventAction::kComputeNode &&
                                   event.op_type == "Inner";
                          }));
  EXPECT_EQ(outer.GetShape("r").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(builder.GetShape("result").Shape(), MakeShape({2, 3}));
}

TEST(GraphBuilder, DefaultDomainFunctionMutationRefreshesTransitiveCallers) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  auto &inner = builder.MakeLocalFunction("Inner");
  inner.SetOpsetVersion("", 18);
  inner.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  inner.MakeNode("Identity", {"a"}, {"r"});
  inner.MakeOutput("r");
  auto &outer = builder.MakeLocalFunction("Outer");
  outer.SetOpsetVersion("", 18);
  outer.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  outer.MakeNode("Inner", {"a"}, {"r"});
  outer.MakeOutput("r");
  builder.MakeNode("Outer", {"x"}, {"result"});

  auto &root_shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
  auto &outer_shapes = const_cast<core::shapes::ShapesContext &>(outer.Shapes());
  root_shapes.set_events_enabled(true);
  outer_shapes.set_events_enabled(true);
  root_shapes.ClearEvents();
  outer_shapes.ClearEvents();
  inner.MakeNode("Neg", {"a"}, {"dead"});

  EXPECT_TRUE(
      std::any_of(root_shapes.Events().begin(), root_shapes.Events().end(), [](const auto &event) {
        return event.action == core::shapes::ShapeEventAction::kComputeNode &&
               event.op_type == "Outer";
      }));
  EXPECT_TRUE(std::any_of(outer_shapes.Events().begin(), outer_shapes.Events().end(),
                          [](const auto &event) {
                            return event.action == core::shapes::ShapeEventAction::kComputeNode &&
                                   event.op_type == "Inner";
                          }));
  EXPECT_EQ(outer.GetShape("r").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(builder.GetShape("result").Shape(), MakeShape({2, 3}));
}

TEST(GraphBuilder, FunctionMutationRefreshesNestedSubgraphOwnerOnce) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("local", 1);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("cond", core::symbolic::TensorType::kBool, MakeShape({}));

  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");

  auto &body = builder.MakeSubgraph("body");
  body.SetOpsetVersion("local", 1);
  body.MakeNode("F", {"x"}, {"body_result"}, "local");
  body.MakeOutput("body_result");
  auto &else_body = builder.MakeSubgraph("else_body");
  else_body.MakeNode("Identity", {"x"}, {"else_result"});
  else_body.MakeOutput("else_result");
  utils::RepeatedProtoField<AttributeProto> attributes;
  auto &then_ref = attributes.add();
  then_ref.set_name("then_branch_ref");
  then_ref.set_type(AttributeProto::STRING);
  then_ref.set_s("body");
  auto &else_ref = attributes.add();
  else_ref.set_name("else_branch_ref");
  else_ref.set_type(AttributeProto::STRING);
  else_ref.set_s("else_body");
  builder.MakeNode("If", {"cond"}, {"result"}, "", "", attributes);

  auto &root_shapes = const_cast<core::shapes::ShapesContext &>(builder.Shapes());
  auto &body_shapes = const_cast<core::shapes::ShapesContext &>(body.Shapes());
  root_shapes.set_events_enabled(true);
  body_shapes.set_events_enabled(true);
  root_shapes.ClearEvents();
  body_shapes.ClearEvents();
  function.MakeNode("Neg", {"a"}, {"dead"});

  const auto count_computations = [](const core::shapes::ShapesContext &shapes,
                                     const std::string &op_type) {
    return std::count_if(shapes.Events().begin(), shapes.Events().end(), [&](const auto &event) {
      return event.action == core::shapes::ShapeEventAction::kComputeNode &&
             event.op_type == op_type;
    });
  };
  EXPECT_EQ(count_computations(body_shapes, "F"), 1);
  EXPECT_EQ(count_computations(root_shapes, "If"), 1);
}

TEST(GraphBuilder, MaintainsConstantInfoIncrementally) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  TensorProto initializer;
  initializer.set_name("weight");
  initializer.set_data_type(TensorProto::DataType::FLOAT);
  initializer.add_dims(1);
  initializer.add_float_data(1.0f);
  builder.MakeInitializer(initializer);
  EXPECT_TRUE(builder.Compute().IsConstantValue("weight"));

  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &value = attributes.add();
  value.set_name("value_float");
  value.set_type(AttributeProto::AttributeType::FLOAT);
  value.set_f(1.0f);
  const std::vector<std::string> outputs =
      builder.MakeNode("Constant", {}, {"constant"}, "", "", attributes);
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_TRUE(builder.Compute().IsConstantValue(outputs[0]));
  EXPECT_TRUE(builder.Compute().NodeConstant(0));
  EXPECT_EQ(builder.Compute().NodeConstant().at(0), core::compute::ConstantInfo::kConstant);
}

TEST(GraphBuilder, MakeInitializerMovePreservesOwnedAllocation) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  TensorProto tensor;
  tensor.set_name("weight");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.add_dims(1024);
  tensor.set_raw_data(std::string(1024 * sizeof(float), '\0'));
  const auto *data = tensor.raw_data().data();
  const auto serialized = tensor.SerializeAsString();

  core::builder::GraphBuilder copying("copy", SchemaLookup());
  EXPECT_EQ(copying.MakeInitializer(tensor), "weight");
  EXPECT_EQ(copying.Initializers()[0].raw_data().data(), data);
  EXPECT_EQ(tensor.SerializeAsString(), serialized);

  EXPECT_EQ(builder.MakeInitializerMove(std::move(tensor)), "weight");
  ASSERT_EQ(builder.Initializers().size(), 1u);
  EXPECT_EQ(builder.Initializers()[0].raw_data().data(), data);
  EXPECT_EQ(builder.Initializers()[0].SerializeAsString(), serialized);
  EXPECT_EQ(tensor.SerializeAsString(), TensorProto().SerializeAsString());
  EXPECT_EQ(builder.GetShape("weight").Shape(), MakeShape({1024}));
  EXPECT_EQ(builder.Compute().ValueTags().at("weight"), "weight");
  EXPECT_TRUE(builder.Compute().IsConstantValue("weight"));

  tensor.set_name("reused");
  tensor.set_data_type(TensorProto::DataType::INT64);
  tensor.add_dims(2);
  tensor.add_int64_data(2);
  tensor.add_int64_data(3);
  EXPECT_EQ(builder.MakeInitializerMove(std::move(tensor)), "reused");
  EXPECT_TRUE(builder.GetShape("reused").HasValueAsShape());
  EXPECT_EQ(builder.Initializers()[0].raw_data().data(), data);
}

TEST(GraphBuilder, MakeInitializerMovePreservesExternalData) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  TensorProto tensor;
  tensor.set_name("external");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.add_dims(4);
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);
  auto *entry = tensor.add_external_data();
  entry->set_key("location");
  entry->set_value("missing-weights.bin");
  entry = tensor.add_external_data();
  entry->set_key("offset");
  entry->set_value("16");
  entry = tensor.add_external_data();
  entry->set_key("length");
  entry->set_value("16");
  tensor.set_doc_string("External initializer");
  const auto serialized = tensor.SerializeAsString();
  EXPECT_EQ(builder.MakeInitializerMove(std::move(tensor)), "external");
  EXPECT_EQ(builder.Initializers()[0].SerializeAsString(), serialized);
  EXPECT_FALSE(builder.Initializers()[0].has_raw_data());
  EXPECT_EQ(builder.GetShape("external").Shape(), MakeShape({4}));
  EXPECT_EQ(tensor.SerializeAsString(), TensorProto().SerializeAsString());
}

TEST(GraphBuilder, MakeInitializerMoveRetainsBorrowedOwner) {
  for (bool is_default : {false, true}) {
    int released = 0;
    std::weak_ptr<void> owner;
    {
      core::builder::GraphBuilder builder("g", SchemaLookup());
      TensorProto tensor;
      tensor.set_name(builder.UniqueName("reserved"));
      tensor.set_data_type(TensorProto::DataType::FLOAT);
      tensor.add_dims(32);
      auto bytes = std::make_shared<std::vector<uint8_t>>(32 * sizeof(float), 0);
      const auto *data = bytes->data();
      tensor.set_raw_data_with_deleter(data, bytes->size(), [bytes, &released]() { ++released; });
      owner = tensor.raw_data().owner();
      bytes.reset();
      const auto serialized = tensor.SerializeAsString();
      EXPECT_THROW(builder.MakeInitializerMove(std::move(tensor)), core::builder::BuilderError);
      EXPECT_EQ(tensor.SerializeAsString(), serialized);
      const auto &unchanged = tensor.raw_data();
      EXPECT_EQ(unchanged.data(), data);
      EXPECT_TRUE(unchanged.is_borrowed());
      EXPECT_EQ(released, 0);
      EXPECT_TRUE(builder.Initializers().empty());

      tensor.set_name("weight");
      if (is_default) {
        builder.MakeInput("weight", core::symbolic::TensorType::kFloat, MakeShape({32}));
      }
      builder.MakeInitializerMove(std::move(tensor));
      const auto &payload = builder.Initializers()[0].raw_data();
      EXPECT_EQ(payload.data(), data);
      EXPECT_TRUE(payload.is_borrowed());
      EXPECT_FALSE(owner.owner_before(payload.owner()));
      EXPECT_FALSE(payload.owner().owner_before(owner));
      EXPECT_FALSE(owner.expired());
      EXPECT_EQ(released, 0);
      EXPECT_EQ(tensor.SerializeAsString(), TensorProto().SerializeAsString());
      EXPECT_FALSE(tensor.raw_data().owner());
      EXPECT_EQ(builder.Compute().IsConstantValue("weight"), !is_default);
    }
    EXPECT_EQ(released, 1);
    EXPECT_TRUE(owner.expired());
  }
}

TEST(GraphBuilder, MakeInitializerMovePreservesOverridableDefault) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  ValueInfoProto input;
  input.set_name("x");
  auto *type = input.mutable_type()->mutable_tensor_type();
  type->set_elem_type(TensorProto::DataType::FLOAT);
  type->mutable_shape()->add_dim()->set_dim_param("batch");
  builder.MakeInput(input);
  const auto shape = builder.GetShape("x").Shape();
  const auto tags = builder.Compute().ValueTags();
  TensorProto tensor = MakeInitializer<float>("x", {2}, {1.f, 2.f});
  EXPECT_EQ(builder.MakeInitializerMove(std::move(tensor)), "x");
  EXPECT_EQ(builder.GetShape("x").Shape(), shape);
  EXPECT_EQ(builder.Compute().ValueTags(), tags);
  EXPECT_FALSE(builder.Compute().IsConstantValue("x"));
  EXPECT_EQ(builder.Inputs()[0].SerializeAsString(), input.SerializeAsString());
  EXPECT_EQ(tensor.SerializeAsString(), TensorProto().SerializeAsString());
}

TEST(GraphBuilder, MakeInitializerMoveValidationDoesNotConsumeSource) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  std::vector<TensorProto> invalid = {MakeInitializer<int64_t>("x", {2}, {1, 2}),
                                      MakeInitializer<float>("x", {1, 2}, {1.f, 2.f}),
                                      MakeInitializer<float>("x", {3}, {1.f, 2.f, 3.f}),
                                      MakeInitializer<float>("", {2}, {1.f, 2.f})};
  for (auto &tensor : invalid) {
    const auto serialized = tensor.SerializeAsString();
    EXPECT_THROW(builder.MakeInitializerMove(std::move(tensor)), core::builder::BuilderError);
    EXPECT_EQ(tensor.SerializeAsString(), serialized);
    EXPECT_TRUE(builder.Initializers().empty());
  }

  for (const auto &name : {"x", "weight"}) {
    TensorProto tensor;
    tensor.set_name(name);
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.add_dims(2);
    tensor.set_raw_data(std::string(2 * sizeof(float), '\0'));
    builder.MakeInitializer(tensor);
    const auto serialized = tensor.SerializeAsString();
    const auto *data = tensor.raw_data().data();
    EXPECT_THROW(builder.MakeInitializerMove(std::move(tensor)), core::builder::BuilderError);
    EXPECT_EQ(tensor.SerializeAsString(), serialized);
    EXPECT_EQ(tensor.raw_data().data(), data);
  }
  EXPECT_EQ(builder.Initializers().size(), 2u);
}

TEST(GraphBuilder, MakeNodeUsesProvidedOutputName) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> outputs = builder.MakeNode("Add", {"x", "y"}, {"z"});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0], "z");
}

TEST(GraphBuilder, OmittedOptionalOutputs) {
  const std::vector<std::string> models = {
      R"(
<ir_version: 10, opset_import: ["": 21]>
pool (float[1,1,4,4] X) => (float[1,1,3,3] Y) {
  Y = MaxPool <kernel_shape: ints = [2,2]> (X)
})",
      R"(
<ir_version: 10, opset_import: ["": 21]>
batchnorm (float[1,2,4,4] X, float[2] scale, float[2] B,
           float[2] mean, float[2] var) => (float[1,2,4,4] Y) {
  Y = BatchNormalization (X, scale, B, mean, var)
})",
      R"(
<ir_version: 10, opset_import: ["": 24]>
attention (float[1,2,3,4] Q, float[1,2,3,4] K,
           float[1,2,3,4] V) => (float[1,2,3,4] Y) {
  Y = Attention (Q, K, V)
})",
      R"(
<ir_version: 10, opset_import: ["": 21]>
loss (float[2,3] scores, int64[2] labels) => (float Y) {
  Y = SoftmaxCrossEntropyLoss (scores, labels)
})",
  };
  for (const auto &text : models) {
    ModelProto model;
    ASSERT_TRUE(OnnxParser::Parse(model, text.c_str()).IsOK());
    const auto &node = model.graph().node(0);
    SCOPED_TRACE(node.op_type().value());
    ASSERT_NO_THROW(checker::check_model(model));

    core::builder::GraphBuilder imported(model, SchemaLookup());
    EXPECT_EQ(imported.Nodes()[0].output().size(), 1u);
    EXPECT_NO_THROW(checker::check_model(imported.ToModel()));

    const auto schemas = SchemaLookup()(node.op_type().value());
    const core::schema::LightOpSchema *schema = nullptr;
    for (const auto &candidate : schemas) {
      if (candidate.since_version() <= model.opset_import(0).version() &&
          (schema == nullptr || candidate.since_version() > schema->since_version())) {
        schema = &candidate;
      }
    }
    ASSERT_NE(schema, nullptr);
    EXPECT_EQ(schema->min_output(), 1);
    NodeProto missing_output = node;
    missing_output.clear_output();
    EXPECT_THROW(schema->Verify(missing_output), core::schema::SchemaError);

    ModelProto too_many_outputs = model;
    auto *invalid_node = too_many_outputs.mutable_graph()->mutable_node(0);
    std::vector<std::string> invalid_outputs = {"Y"};
    for (int i = 0; i < schema->max_output(); ++i) {
      invalid_outputs.push_back("extra" + std::to_string(i));
      invalid_node->add_output(invalid_outputs.back());
    }
    EXPECT_THROW(schema->Verify(*invalid_node), core::schema::SchemaError);
    EXPECT_THROW(core::builder::GraphBuilder(too_many_outputs, SchemaLookup()),
                 core::builder::BuilderError);

    for (bool automatic_outputs : {false, true}) {
      core::builder::GraphBuilder builder("optional_outputs", SchemaLookup());
      builder.SetOpsetVersion("", model.opset_import(0).version());
      for (const auto &input : model.graph().input()) {
        builder.MakeInput(input);
      }
      std::vector<std::string> inputs;
      for (const auto &input : node.input()) {
        inputs.push_back(input.value());
      }
      EXPECT_THROW(builder.MakeNode(node.op_type().value(), inputs, invalid_outputs, "", "",
                                    node.attribute()),
                   core::builder::BuilderError);
      const auto outputs = builder.MakeNode(node.op_type().value(), inputs,
                                            automatic_outputs ? std::vector<std::string>{}
                                                              : std::vector<std::string>{"Y"},
                                            "", "", node.attribute());
      ASSERT_EQ(outputs.size(), 1u);
      builder.MakeOutput(outputs[0]);
      EXPECT_NO_THROW(checker::check_model(builder.ToModel()));
    }
  }
}

TEST(GraphBuilder, MakeNodeRejectsUnknownInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  EXPECT_THROW(builder.MakeNode("Add", {"missing", "y"}), core::builder::BuilderError);
}

TEST(GraphBuilder, MakeNodeAcceptsRepeatedProtoFieldAttributes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &alpha = attributes.add();
  alpha.set_name("alpha");
  alpha.set_type(AttributeProto::AttributeType::FLOAT);
  alpha.set_f(0.25f);

  const std::vector<std::string> outputs =
      builder.MakeNode("LeakyRelu", {"x"}, {}, "", "", attributes);
  ASSERT_EQ(outputs.size(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  const NodeProto &node = builder.Nodes()[0];
  ASSERT_EQ(node.attribute().size(), 1);
  EXPECT_EQ(node.attribute()[0].name().value(), "alpha");
  EXPECT_FLOAT_EQ(node.attribute()[0].f(), 0.25f);
}

TEST(GraphBuilder, ExternalInitializerIsRecorded) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeExternalInitializer("w", core::symbolic::TensorType::kFloat, {4, 4}, "weights.bin", 0,
                                  64);
  EXPECT_TRUE(builder.HasName("w"));
  ASSERT_EQ(builder.Initializers().size(), 1u);
  const TensorProto &init = builder.Initializers()[0];
  EXPECT_EQ(init.name().value(), "w");
  EXPECT_EQ(init.data_location(), TensorProto::DataLocation::EXTERNAL);
}

TEST(GraphBuilder, ToModelProducesGraphWithOpset) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> outputs = builder.MakeNode("Add", {"x", "y"});
  builder.MakeOutput(outputs[0]);

  const ModelProto model = builder.ToModel();
  EXPECT_GT(model.ir_version(), 0);
  EXPECT_EQ(model.graph().node().size(), 1);
  ASSERT_GT(model.opset_import().size(), 0);
}

TEST(GraphBuilder, BuildsTranslatedLinearModel) {
  core::builder::GraphBuilder builder("linear", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  TensorProto weight;
  weight.set_name("W");
  weight.set_data_type(TensorProto::DataType::FLOAT);
  weight.add_dims(3);
  weight.add_float_data(1.0f);
  weight.add_float_data(2.0f);
  weight.add_float_data(3.0f);
  builder.MakeInitializer(weight);

  TensorProto bias;
  bias.set_name("B");
  bias.set_data_type(TensorProto::DataType::FLOAT);
  bias.add_dims(3);
  bias.add_float_data(0.5f);
  bias.add_float_data(0.5f);
  bias.add_float_data(0.5f);
  builder.MakeInitializer(bias);

  const std::vector<std::string> multiplied = builder.MakeNode("Mul", {"X", "W"}, {"XW"});
  const std::vector<std::string> added = builder.MakeNode("Add", {multiplied[0], "B"}, {"Y"});
  builder.MakeOutput(added[0], core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const ModelProto model = builder.ToModel();
  ASSERT_EQ(model.graph().node().size(), 2);
  EXPECT_EQ(model.graph().node()[0].op_type().value(), "Mul");
  EXPECT_EQ(model.graph().node()[1].op_type().value(), "Add");
  EXPECT_EQ(model.graph().initializer().size(), 2);
  EXPECT_EQ(model.graph().output()[0].name().value(), "Y");
}

TEST(GraphBuilder, ToFunctionMaterializesInitializersWithoutMutatingBuilder) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeInitializer(MakeInitializer<float>("w", {2}, {1.0f, 2.0f}));
  builder.MakeExternalInitializer("external", core::symbolic::TensorType::kFloat, {2}, "w.bin", 0,
                                  8);
  builder.MakeNode("Add", {"x", "w"}, {"middle"});
  builder.MakeNode("Add", {"middle", "external"}, {"y"});
  builder.MakeOutput("y");
  const auto before = builder.ToGraph();
  const auto function = builder.ToFunction("custom");
  ASSERT_EQ(function.node().size(), 4u);
  for (size_t index = 0; index < 2; ++index) {
    const auto &node = function.node(index);
    EXPECT_EQ(node.op_type(), "Constant");
    ASSERT_EQ(node.attribute().size(), 1u);
    EXPECT_EQ(node.attribute(0).name(), "value");
    EXPECT_EQ(node.attribute(0).t().SerializeAsString(),
              builder.Initializers()[index].SerializeAsString());
    EXPECT_EQ(node.output(0), builder.Initializers()[index].name().value());
    EXPECT_EQ(function.node(index + 2).SerializeAsString(), before.node(index).SerializeAsString());
  }
  EXPECT_EQ(builder.ToGraph().SerializeAsString(), before.SerializeAsString());
  EXPECT_EQ(builder.ToFunction("custom").SerializeAsString(), function.SerializeAsString());
}

TEST(GraphBuilder, ToFunctionRejectsOverridableInitializersAndMissingOpset) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}});
  builder.MakeInput("w", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeInitializer(MakeInitializer<float>("w", {2}, {1.0f, 2.0f}));
  builder.MakeOutput("w");
  const auto before = builder.ToGraph().SerializeAsString();
  EXPECT_THROW(builder.ToFunction("custom"), core::builder::BuilderError);
  EXPECT_EQ(builder.ToGraph().SerializeAsString(), before);

  core::builder::GraphBuilder missing_opset("constant", SchemaLookup());
  missing_opset.MakeInitializer(MakeInitializer<float>("w", {2}, {1.0f, 2.0f}));
  missing_opset.MakeOutput("w");
  EXPECT_THROW(missing_opset.ToFunction("custom"), core::builder::BuilderError);
}

TEST(GraphBuilder, ToFunctionValidatesInitializerTypesAtSelectedOpset) {
  for (int opset : {8, 9}) {
    for (bool optimize : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "opset=" << opset << ", optimize=" << optimize);
      core::builder::GraphBuilder builder("reshape", SchemaLookup(), {{"", opset}, {"custom", 1}});
      builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
      const auto dims = MakeInitializer<int64_t>("dims", {2}, {3, 2});
      builder.MakeInitializer(dims);
      builder.MakeNode("Reshape", {"x", "dims"}, {"y"});
      builder.MakeOutput("y");
      if (optimize) {
        core::builder::GraphGraph graph(builder);
        graph.Optimize();
      }
      const auto before = builder.BuildGraph().SerializeAsString();
      const auto opsets = builder.OpsetVersions();
      const auto shape = builder.GetShape("y").Shape();
      if (opset == 8) {
        try {
          builder.ToFunction("custom");
          FAIL() << "INT64 initializer must not become Constant-1";
        } catch (const core::builder::BuilderError &error) {
          const std::string message = error.what();
          EXPECT_NE(message.find("dims"), std::string::npos);
          EXPECT_NE(message.find("INT64"), std::string::npos);
          EXPECT_NE(message.find("opset 8"), std::string::npos);
        }
      } else {
        const auto function = builder.ToFunction("custom");
        ASSERT_EQ(function.node().size(), 2u);
        EXPECT_EQ(function.node(0).op_type(), "Constant");
        EXPECT_EQ(function.node(0).output(0), "dims");
        EXPECT_EQ(function.node(0).attribute(0).t().SerializeAsString(), dims.SerializeAsString());
        ASSERT_EQ(function.opset_import().size(), opsets.size());
        for (const auto &entry : function.opset_import())
          EXPECT_EQ(entry.version(), builder.OpsetVersion(entry.domain().value()));
      }
      EXPECT_EQ(builder.BuildGraph().SerializeAsString(), before);
      EXPECT_EQ(builder.OpsetVersions(), opsets);
      EXPECT_EQ(builder.GetShape("y").Shape(), shape);
    }
  }
}

TEST(GraphBuilder, ToFunctionAllowsFloatingInitializerAtOpset8) {
  core::builder::GraphBuilder builder("add", SchemaLookup(), {{"", 8}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  const auto weight = MakeInitializer<float>("w", {2}, {1.f, 2.f});
  builder.MakeInitializer(weight);
  builder.MakeNode("Add", {"x", "w"}, {"y"});
  builder.MakeOutput("y");
  const auto function = builder.ToFunction("custom");
  ASSERT_EQ(function.node().size(), 2u);
  EXPECT_EQ(function.node(0).op_type(), "Constant");
  EXPECT_EQ(function.node(0).attribute(0).t().SerializeAsString(), weight.SerializeAsString());
  ASSERT_EQ(function.opset_import().size(), 1u);
  EXPECT_EQ(function.opset_import(0).version(), 8);
  EXPECT_EQ(builder.OpsetVersion(""), 8);
}

TEST(GraphBuilder, LocalFunctionDefinitionValidatesInitializerConstantTypes) {
  for (int opset : {8, 9}) {
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", opset}, {"custom", 1}});
    auto &function = builder.MakeLocalFunction("F", "custom");
    function.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    const auto dims = MakeInitializer<int64_t>("dims", {2}, {3, 2});
    function.MakeInitializer(dims);
    function.MakeNode("Reshape", {"x", "dims"}, {"y"});
    const auto opsets = function.OpsetVersions();
    if (opset == 8) {
      EXPECT_THROW(function.MakeOutput("y"), core::builder::BuilderError);
      EXPECT_THROW(builder.ToModel(), core::builder::BuilderError);
    } else {
      function.MakeOutput("y");
      const auto model = builder.ToModel();
      ASSERT_EQ(model.functions().size(), 1u);
      const auto &exported = model.functions(0);
      ASSERT_EQ(exported.node().size(), 2u);
      EXPECT_EQ(exported.node(0).op_type(), "Constant");
      EXPECT_EQ(exported.node(0).attribute(0).t().SerializeAsString(), dims.SerializeAsString());
      for (const auto &entry : exported.opset_import())
        EXPECT_EQ(entry.version(), function.OpsetVersion(entry.domain().value()));
    }
    EXPECT_EQ(function.OpsetVersions(), opsets);
    EXPECT_EQ(builder.OpsetVersions(), opsets);
    ASSERT_EQ(function.Initializers().size(), 1u);
    EXPECT_EQ(function.Initializers()[0].SerializeAsString(), dims.SerializeAsString());
  }
}

TEST(GraphBuilder, LocalFunctionWithInitializerPreservesShapeInferenceAndModelExport) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}, {"custom", 1}});
  auto &function = builder.MakeLocalFunction("F", "custom");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2}));
  function.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  function.MakeInitializer(MakeInitializer<float>("w", {2}, {1.0f, 2.0f}));
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const auto &name : {"then_branch", "else_branch"}) {
    auto &branch = function.MakeSubgraph(name);
    branch.MakeNode("Add", {"a", "w"}, {"result"});
    branch.MakeOutput("result");
    branch.set_device(core::symbolic::Device::kGPU0);
    auto &reference = attributes.add();
    reference.set_name(std::string(name) + "_ref");
    reference.set_type(AttributeProto::STRING);
    reference.set_s(name);
  }
  function.MakeNode("If", {"condition"}, {"b"}, "", "", attributes);
  function.MakeOutput("b");
  function.set_device(core::symbolic::Device::kCPU);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeNode("F", {"x", "condition"}, {"y"}, "custom");
  builder.MakeOutput("y");
  EXPECT_EQ(builder.GetShape("y").Shape(), MakeShape({2}));
  const auto model = builder.ToModel();
  ASSERT_EQ(model.functions().size(), 1u);
  ASSERT_EQ(model.functions(0).node().size(), 2u);
  EXPECT_EQ(model.functions(0).node(0).op_type(), "Constant");
  EXPECT_EQ(model.functions(0).node(1).op_type(), "If");
  EXPECT_NO_THROW(VerifyModel(model));
  core::builder::GraphBuilder copy(model, SchemaLookup());
  copy.CopyDeviceTargetsFrom(builder);
  EXPECT_EQ(copy.LocalFunction("F").device(), core::symbolic::Device::kCPU);
  for (const auto &branch : copy.LocalFunction("F").Subgraphs())
    EXPECT_EQ(branch->device(), core::symbolic::Device::kGPU0);
}

TEST(GraphBuilder, ToFunctionPreservesFinalizedShapesAndMetadata) {
  core::builder::GraphBuilder builder("F", SchemaLookup(), {{"", 28}});
  ValueInfoProto input;
  input.set_name("X");
  input.set_doc_string("Input declaration");
  auto *input_type = input.mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(TensorProto::DataType::FLOAT);
  input_type->mutable_shape()->add_dim()->set_dim_param("");
  input_type->mutable_shape()->add_dim()->set_dim_value(3);
  builder.MakeInput(input);
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeNode("Neg", {"X"}, {"intermediate"});
  builder.MakeNode("Abs", {"intermediate"}, {"out"});
  builder.MakeNode("Shape", {"intermediate"}, {"shape"});
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeNode("Identity", {"intermediate"}, {name + "_result"});
    branch.MakeOutput(name + "_result");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  for (const std::string name : {"out", "shape", "selected"}) {
    builder.MakeOutput(name);
  }

  const FunctionProto function = builder.ToFunction("custom");
  const GraphProto graph = builder.ToGraph();
  EXPECT_EQ(function.name(), "F");
  EXPECT_EQ(function.domain(), "custom");
  ASSERT_EQ(function.node().size(), graph.node().size());
  for (std::size_t i = 0; i < graph.node().size(); ++i) {
    EXPECT_EQ(function.node(i).SerializeAsString(), graph.node(i).SerializeAsString());
  }
  EXPECT_EQ(function.node(1).op_type(), "Shape");
  ASSERT_EQ(function.value_info().size(),
            graph.input().size() + graph.output().size() + graph.value_info().size());
  const auto check_value_info = [&](const auto &values) {
    for (const ValueInfoProto &expected : values) {
      bool found = false;
      for (const ValueInfoProto &actual : function.value_info()) {
        if (actual.name() == expected.name()) {
          EXPECT_EQ(actual.SerializeAsString(), expected.SerializeAsString());
          found = true;
          break;
        }
      }
      EXPECT_TRUE(found) << expected.name();
    }
  };
  check_value_info(graph.input());
  check_value_info(graph.output());
  check_value_info(graph.value_info());
  EXPECT_EQ(builder.ToFunction("custom").SerializeAsString(), function.SerializeAsString());
}

TEST(GraphBuilder, ExportFunctionNormalizesDefaultNodeDomainsRecursively) {
  ModelProto model;
  model.add_opset("ai.onnx", 18);
  model.add_opset("local", 1);
  FunctionProto &function = *model.add_functions();
  function.set_name("F");
  function.set_domain("local");
  function.add_opset("ai.onnx", 18);
  function.add_input("x");
  function.add_input("condition");
  function.add_output("y");
  auto *if_node = function.add_node();
  if_node->set_domain("ai.onnx");
  if_node->set_op_type("If");
  if_node->add_input("condition");
  if_node->add_output("y");
  for (const std::string branch_name : {"then_branch", "else_branch"}) {
    auto *attribute = if_node->add_attribute();
    attribute->set_name(branch_name);
    attribute->set_type(AttributeProto::GRAPH);
    auto *branch = attribute->mutable_g();
    auto *identity = branch->add_node();
    identity->set_domain("ai.onnx");
    identity->set_op_type("Identity");
    identity->add_input("x");
    identity->add_output("branch_output");
    auto *output = branch->add_output();
    output->set_name("branch_output");
    auto *type = output->mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(2);
  }
  const auto add_input = [&](const std::string &name, TensorProto::DataType dtype, bool scalar) {
    auto *input = function.add_value_info();
    input->set_name(name);
    auto *type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(dtype);
    if (scalar)
      type->mutable_shape();
    else
      type->mutable_shape()->add_dim()->set_dim_value(2);
  };
  add_input("x", TensorProto::FLOAT, false);
  add_input("condition", TensorProto::BOOL, true);
  auto *output_info = function.add_value_info();
  output_info->set_name("y");
  auto *output_type = output_info->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(TensorProto::FLOAT);
  output_type->mutable_shape()->add_dim()->set_dim_value(2);

  GraphProto &graph = *model.mutable_graph();
  auto *x = graph.add_input();
  x->set_name("x");
  *x->mutable_type() = function.value_info(0).type();
  auto *condition = graph.add_input();
  condition->set_name("condition");
  *condition->mutable_type() = function.value_info(1).type();
  auto *call = graph.add_node();
  call->set_domain("local");
  call->set_op_type("F");
  call->add_input("x");
  call->add_input("condition");
  call->add_output("y");
  auto *output = graph.add_output();
  output->set_name("y");
  *output->mutable_type() = function.value_info(2).type();

  core::builder::GraphBuilder builder(model, SchemaLookup());
  const ModelProto exported = builder.ToModel();
  ASSERT_EQ(exported.functions().size(), 1u);
  const FunctionProto &exported_function = exported.functions(0);
  EXPECT_TRUE(std::any_of(
      exported_function.opset_import().begin(), exported_function.opset_import().end(),
      [](const auto &opset) { return opset.domain().empty() && opset.version() == 18; }));
  EXPECT_FALSE(std::any_of(exported_function.opset_import().begin(),
                           exported_function.opset_import().end(),
                           [](const auto &opset) { return opset.domain() == "ai.onnx"; }));
  ASSERT_EQ(exported_function.node().size(), 1u);
  EXPECT_TRUE(exported_function.node(0).domain().empty());
  for (const auto &attribute : exported_function.node(0).attribute()) {
    ASSERT_TRUE(attribute.has_g());
    ASSERT_EQ(attribute.g().node().size(), 1u);
    EXPECT_TRUE(attribute.g().node(0).domain().empty());
  }
  EXPECT_NO_THROW(VerifyModel(exported));
}

TEST(GraphBuilder, ExplicitOpsetIsPreserved) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 17);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Add", {"x", "y"});
  EXPECT_EQ(builder.OpsetVersion(""), 17);
}

TEST(GraphBuilder, InputOutputFromProto) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  ValueInfoProto vi;
  vi.set_name("x");
  core::symbolic::SymTensorToValueInfo(
      core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, MakeShape({2, 3})),
      vi);
  builder.MakeInput(vi);
  EXPECT_TRUE(builder.HasName("x"));
  ASSERT_EQ(builder.Inputs().size(), 1u);
  EXPECT_EQ(builder.Inputs()[0].name().value(), "x");
  EXPECT_TRUE(builder.HasShape("x"));

  ValueInfoProto out;
  out.set_name("x");
  builder.MakeOutput(out);
  ASSERT_EQ(builder.Outputs().size(), 1u);
  EXPECT_EQ(builder.Outputs()[0].name().value(), "x");
}

TEST(GraphBuilder, ToStringDescribesContent) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> outputs = builder.MakeNode("Add", {"x", "y"});
  builder.MakeOutput(outputs[0]);

  const std::string text = builder.ToString();
  EXPECT_NE(text.find("GraphBuilder(name=g)"), std::string::npos);
  EXPECT_NE(text.find("inputs (2)"), std::string::npos);
  EXPECT_NE(text.find("nodes (1)"), std::string::npos);
  EXPECT_NE(text.find("Add("), std::string::npos);
}

TEST(GraphBuilder, LocalFunctionIsANestedBuilder) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("MyFct", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  fct.MakeInput("b", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> outputs = fct.MakeNode("Add", {"a", "b"});
  fct.MakeOutput(outputs[0]);

  EXPECT_TRUE(builder.HasLocalFunction("MyFct"));
  EXPECT_EQ(builder.LocalFunction("MyFct").Nodes().size(), 1u);
  EXPECT_THROW(builder.MakeLocalFunction("MyFct"), core::builder::BuilderError);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("z", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> top = builder.MakeNode("MyFct", {"x", "z"}, {}, "custom");
  builder.MakeOutput(top[0]);

  const ModelProto model = builder.ToModel();
  ASSERT_EQ(model.functions().size(), 1);
  EXPECT_EQ(model.functions(0).name().value(), "MyFct");
}

TEST(GraphBuilder, SubgraphIsANestedBuilder) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  EXPECT_TRUE(builder.HasSubgraph("body"));
  EXPECT_EQ(builder.Subgraph("body").Inputs().size(), 1u);
  EXPECT_THROW(builder.MakeSubgraph("body"), core::builder::BuilderError);
}

TEST(GraphBuilder, RemoveUnusedNodesDropsDeadEnds) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const std::vector<std::string> used = builder.MakeNode("Add", {"x", "y"});
  // Two chained nodes that no graph output depends on: both are dead ends.
  const std::vector<std::string> dead = builder.MakeNode("Mul", {"x", "y"});
  builder.MakeNode("Neg", {dead[0]});
  builder.MakeOutput(used[0]);

  EXPECT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.RemoveUnusedNodes(), 2u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Add");
  // A second pass has nothing left to remove.
  EXPECT_EQ(builder.RemoveUnusedNodes(), 0u);
}

TEST(GraphBuilder, RemoveUnusedNodesKeepsSharedProducer) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // A shared producer feeding one used and one unused consumer must be kept.
  const std::vector<std::string> shared = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> used = builder.MakeNode("Abs", {shared[0]});
  builder.MakeNode("Abs", {shared[0]}); // dead-end consumer of the shared value
  builder.MakeOutput(used[0]);

  EXPECT_EQ(builder.RemoveUnusedNodes(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Abs");
}

TEST(GraphBuilder, RewritingPreservesTagsWhenConsumerIsRemoved) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInitializer(MakeInitializer<int64_t>("dims", {2}, {3, 2}));
  builder.MakeNode("Reshape", {"X", "dims"}, {"unused"});
  builder.MakeNode("Neg", {"dims"}, {"neg_dims"});
  builder.MakeOutput("dims");
  builder.MakeOutput("neg_dims");

  EXPECT_EQ(builder.Compute().ValueTags().at("dims"), "shape");
  EXPECT_EQ(builder.Compute().ValueTags().at("neg_dims"), "shape");
  EXPECT_EQ(builder.Compute().NodeTag(1), "shape");

  RewritingProto plan;
  plan.add_rewrites()->add_matched_nodes(0);
  const auto rewritten = builder.ApplyRewritings(plan);
  ASSERT_EQ(rewritten.Nodes().size(), 1u);
  EXPECT_EQ(rewritten.Nodes()[0].op_type().value(), "Neg");
  ASSERT_EQ(rewritten.Initializers().size(), 1u);
  EXPECT_EQ(rewritten.Initializers()[0].name().value(), "dims");
  EXPECT_EQ(rewritten.Compute().ValueTags().at("dims"), "shape");
  EXPECT_EQ(rewritten.Compute().ValueTags().at("neg_dims"), "shape");
  EXPECT_EQ(rewritten.Compute().NodeTag(0), "shape");

  // Fresh inference on the untagged reduced graph is not the rewrite's tag contract.
  const auto fresh_tags = core::compute::InferValueAndNodeTags(rewritten.BuildGraph());
  EXPECT_EQ(fresh_tags.first.at("dims"), "weight");
  EXPECT_EQ(fresh_tags.first.at("neg_dims"), "weight");
}

TEST(GraphBuilder, RemoveUnusedNodesRecursesIntoSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // A nested subgraph with one live and one dead-end node.
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> body_used = body.MakeNode("Neg", {"a"});
  body.MakeNode("Abs", {"a"}); // dead end inside the subgraph
  body.MakeOutput(body_used[0]);

  const std::vector<std::string> used = builder.MakeNode("Identity", {"x"});
  builder.MakeOutput(used[0]);

  // The removal descends into the subgraph and prunes its dead-end node.
  EXPECT_EQ(builder.RemoveUnusedNodes(), 1u);
  EXPECT_EQ(builder.Nodes().size(), 1u);
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 1u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].op_type().value(), "Neg");
}

TEST(GraphBuilder, RemoveDuplicateInitializersCollapsesEqual) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));

  // Two initializers with byte-for-byte identical content.
  builder.MakeInitializer(MakeInitializer<float>("w1", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
  builder.MakeInitializer(MakeInitializer<float>("w2", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));

  const std::vector<std::string> a = builder.MakeNode("Add", {"x", "w1"});
  const std::vector<std::string> b = builder.MakeNode("Add", {"x", "w2"});
  builder.MakeOutput(a[0]);
  builder.MakeOutput(b[0]);

  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 1u);
  ASSERT_EQ(builder.Initializers().size(), 1u);
  EXPECT_EQ(builder.Initializers()[0].name().value(), "w1");
  // The reference to the dropped duplicate is rewritten to the surviving name.
  EXPECT_EQ(builder.Nodes()[1].input(1), "w1");
  // A second pass has nothing left to collapse.
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 0u);
  const GraphProto graph = builder.ToGraph();
  EXPECT_EQ(builder.ToGraph().SerializeAsString(), graph.SerializeAsString());
  core::compute::ComputeContext expected;
  const auto tags = expected.ComputeValueAndNodeTags(graph);
  expected.ComputeInPlaceReuseGraph(graph, builder.Shapes(), false, tags.first);
  expected.ComputePeakMemory(graph, builder.device());
  EXPECT_EQ(builder.Compute().Reuse(), expected.Reuse());
  EXPECT_EQ(builder.Compute().Memory(), expected.Memory());
  EXPECT_EQ(builder.Compute().PeakMemory(), expected.PeakMemory());
}

TEST(GraphBuilder, RemoveDuplicateInitializersKeepsDistinctContent) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));

  // Same shape and type but different data: not duplicates.
  builder.MakeInitializer(MakeInitializer<float>("w1", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));
  builder.MakeInitializer(MakeInitializer<float>("w2", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f}));

  const std::vector<std::string> a = builder.MakeNode("Add", {"x", "w1"});
  const std::vector<std::string> b = builder.MakeNode("Add", {"x", "w2"});
  builder.MakeOutput(a[0]);
  builder.MakeOutput(b[0]);

  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 0u);
  EXPECT_EQ(builder.Initializers().size(), 2u);
}

TEST(GraphBuilder, RemoveDuplicateInitializersRecursesIntoSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));
  body.MakeInitializer(MakeInitializer<float>("c1", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  body.MakeInitializer(MakeInitializer<float>("c2", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  const std::vector<std::string> s = body.MakeNode("Add", {"a", "c2"});
  body.MakeOutput(s[0]);

  // The dedup descends into the subgraph and collapses its duplicate.
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 1u);
  ASSERT_EQ(builder.Subgraph("body").Initializers().size(), 1u);
  EXPECT_EQ(builder.Subgraph("body").Initializers()[0].name().value(), "c1");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].input(1), "c1");
}

TEST(GraphBuilder, RemoveDuplicateInitializersCollapsesAcrossScopes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));

  // An initializer in the enclosing graph.
  builder.MakeInitializer(MakeInitializer<float>("w", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  const std::vector<std::string> a = builder.MakeNode("Add", {"x", "w"});
  builder.MakeOutput(a[0]);

  // The subgraph declares its own initializer with the same content. Because a
  // subgraph body sees the enclosing scope, the duplicate is collapsed onto the
  // enclosing initializer and its reference is rewritten to "w".
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("b", core::symbolic::TensorType::kFloat, MakeShape({2, 2}));
  body.MakeInitializer(MakeInitializer<float>("c", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}));
  const std::vector<std::string> s = body.MakeNode("Add", {"b", "c"});
  body.MakeOutput(s[0]);

  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 1u);
  ASSERT_EQ(builder.Initializers().size(), 1u);
  EXPECT_EQ(builder.Initializers()[0].name().value(), "w");
  EXPECT_EQ(builder.Subgraph("body").Initializers().size(), 0u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].input(1), "w");
}

TEST(GraphBuilder, RemoveIdentityNodesRewiresConsumers) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // Identity forwards x; a downstream node consumes the forwarded value.
  const std::vector<std::string> ident = builder.MakeNode("Identity", {"x"});
  const std::vector<std::string> used = builder.MakeNode("Neg", {ident[0]});
  builder.MakeOutput(used[0]);

  EXPECT_EQ(builder.RemoveIdentityNodes(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  // The consumer now reads directly from the identity's input.
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  // A second pass has nothing left to remove.
  EXPECT_EQ(builder.RemoveIdentityNodes(), 0u);
}

TEST(GraphBuilder, RemoveIdentityNodesCollapsesChains) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // A chain of two identities feeding a live consumer must collapse onto x.
  const std::vector<std::string> i1 = builder.MakeNode("Identity", {"x"});
  const std::vector<std::string> i2 = builder.MakeNode("Identity", {i1[0]});
  const std::vector<std::string> used = builder.MakeNode("Neg", {i2[0]});
  builder.MakeOutput(used[0]);

  EXPECT_EQ(builder.RemoveIdentityNodes(), 2u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
}

TEST(GraphBuilder, RemoveIdentityNodesKeepsGraphOutputIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // The identity output is a declared graph output, so the node is kept.
  const std::vector<std::string> ident = builder.MakeNode("Identity", {"x"});
  builder.MakeOutput(ident[0]);

  EXPECT_EQ(builder.RemoveIdentityNodes(), 0u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Identity");
}

TEST(GraphBuilder, RemoveIdentityNodesRecursesIntoSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // A nested subgraph with an interior identity to collapse.
  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> inner = body.MakeNode("Identity", {"a"});
  const std::vector<std::string> body_used = body.MakeNode("Neg", {inner[0]});
  body.MakeOutput(body_used[0]);

  const std::vector<std::string> top = builder.MakeNode("Neg", {"x"});
  builder.MakeOutput(top[0]);

  EXPECT_EQ(builder.RemoveIdentityNodes(), 1u);
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 1u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].input(0), "a");
}

TEST(GraphBuilder, RemoveDuplicateNodesCollapsesEqual) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // Two Neg nodes computing the same value feed a shared consumer.
  const std::vector<std::string> a = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> b = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> sum = builder.MakeNode("Add", {a[0], b[0]});
  builder.MakeOutput(sum[0]);

  EXPECT_EQ(builder.RemoveDuplicateNodes(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Add");
  // Both Add inputs now read from the surviving Neg.
  EXPECT_EQ(builder.Nodes()[1].input(0), a[0]);
  EXPECT_EQ(builder.Nodes()[1].input(1), a[0]);
  // A second pass has nothing left to collapse.
  EXPECT_EQ(builder.RemoveDuplicateNodes(), 0u);
}

TEST(GraphBuilder, RemoveDuplicateNodesCollapsesBranches) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // Two identical branches: once the first level collapses, the second level
  // becomes duplicated too and collapses in the same pass.
  const std::vector<std::string> a1 = builder.MakeNode("Mul", {"x", "x"});
  const std::vector<std::string> b1 = builder.MakeNode("Mul", {"x", "x"});
  const std::vector<std::string> a2 = builder.MakeNode("Relu", {a1[0]});
  const std::vector<std::string> b2 = builder.MakeNode("Relu", {b1[0]});
  const std::vector<std::string> sum = builder.MakeNode("Add", {a2[0], b2[0]});
  builder.MakeOutput(sum[0]);

  EXPECT_EQ(builder.RemoveDuplicateNodes(), 2u);
  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Mul");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Relu");
  EXPECT_EQ(builder.Nodes()[2].op_type().value(), "Add");
  EXPECT_EQ(builder.Nodes()[2].input(0), a2[0]);
  EXPECT_EQ(builder.Nodes()[2].input(1), a2[0]);
}

TEST(GraphBuilder, RemoveDuplicateNodesKeepsDistinctAttributes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const auto leaky = [&](float alpha) {
    utils::RepeatedProtoField<AttributeProto> attributes;
    AttributeProto &attr = attributes.add();
    attr.set_name("alpha");
    attr.set_type(AttributeProto::AttributeType::FLOAT);
    attr.set_f(alpha);
    return builder.MakeNode("LeakyRelu", {"x"}, {}, "", "", attributes);
  };
  const std::vector<std::string> a = leaky(0.1f);
  const std::vector<std::string> b = leaky(0.2f);
  const std::vector<std::string> sum = builder.MakeNode("Add", {a[0], b[0]});
  builder.MakeOutput(sum[0]);

  // Different attribute values mean different computations: nothing collapses.
  EXPECT_EQ(builder.RemoveDuplicateNodes(), 0u);
  ASSERT_EQ(builder.Nodes().size(), 3u);
}

TEST(GraphBuilder, RemoveDuplicateNodesChecksNativeInputsOnSignatureCollision) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  const std::string joined = "a\x1e"
                             "b";
  for (const auto &name : {std::string("a"), std::string("b"), joined})
    builder.MakeInput(name, core::symbolic::TensorType::kFloat, MakeShape({2}));
  const auto first = builder.MakeNode("Sum", {joined});
  const auto second = builder.MakeNode("Sum", {"a", "b"});
  const auto sum = builder.MakeNode("Add", {first[0], second[0]});
  builder.MakeOutput(sum[0]);
  ASSERT_EQ(builder.Nodes()[0].Signature({joined}), builder.Nodes()[1].Signature({"a", "b"}));
  EXPECT_EQ(builder.RemoveDuplicateNodes(), 0u);
  ASSERT_EQ(builder.Nodes().size(), 3u);
}

TEST(GraphBuilder, RemoveDuplicateNodesIgnoresAttributeOrderButChecksEveryValue) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const auto make = [&](bool reverse, float maximum) {
    utils::RepeatedProtoField<AttributeProto> attributes;
    for (bool high : {reverse, !reverse}) {
      auto &attribute = attributes.add();
      attribute.set_name(high ? "beta" : "alpha");
      attribute.set_type(AttributeProto::FLOAT);
      attribute.set_f(high ? maximum : 0.0f);
    }
    return builder.MakeNode("HardSigmoid", {"x"}, {}, "", "", attributes);
  };
  const auto first = make(false, 1.0f);
  const auto second = make(true, 1.0f);
  const auto third = make(false, 2.0f);
  const auto sum = builder.MakeNode("Sum", {first[0], second[0], third[0]});
  builder.MakeOutput(sum[0]);
  EXPECT_EQ(builder.RemoveDuplicateNodes(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[2].input(0), builder.Nodes()[2].input(1));
  EXPECT_NE(builder.Nodes()[2].input(0), builder.Nodes()[2].input(2));
}

TEST(GraphBuilder, RemoveDuplicateNodesKeepsGraphOutputs) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // Both duplicates are declared graph outputs, so neither can be dropped.
  const std::vector<std::string> a = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> b = builder.MakeNode("Neg", {"x"});
  builder.MakeOutput(a[0]);
  builder.MakeOutput(b[0]);

  EXPECT_EQ(builder.RemoveDuplicateNodes(), 0u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
}

TEST(GraphBuilder, RemoveDuplicateNodesRecursesIntoSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> n1 = body.MakeNode("Neg", {"a"});
  const std::vector<std::string> n2 = body.MakeNode("Neg", {"a"});
  const std::vector<std::string> s = body.MakeNode("Add", {n1[0], n2[0]});
  body.MakeOutput(s[0]);

  const std::vector<std::string> top = builder.MakeNode("Neg", {"x"});
  builder.MakeOutput(top[0]);

  EXPECT_EQ(builder.RemoveDuplicateNodes(), 1u);
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 2u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[1].op_type().value(), "Add");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[1].input(0), n1[0]);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[1].input(1), n1[0]);
}

TEST(GraphBuilder, MoveShapeAndSizeNodesHoistsToProducer) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  const std::vector<std::string> p = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> a = builder.MakeNode("Add", {p[0], p[0]});
  // Shape reads p but is inserted after an unrelated node; the chained Size
  // reads the Shape output, so both must move right after Neg in one pass.
  const std::vector<std::string> sh = builder.MakeNode("Shape", {p[0]});
  const std::vector<std::string> sz = builder.MakeNode("Size", {sh[0]});
  builder.MakeOutput(a[0]);
  builder.MakeOutput(sz[0]);

  EXPECT_EQ(builder.MoveShapeAndSizeNodes(), 2u);
  ASSERT_EQ(builder.Nodes().size(), 4u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Shape");
  EXPECT_EQ(builder.Nodes()[2].op_type().value(), "Size");
  EXPECT_EQ(builder.Nodes()[3].op_type().value(), "Add");
  // A second pass leaves the already-tightened graph untouched.
  EXPECT_EQ(builder.MoveShapeAndSizeNodes(), 0u);
}

TEST(GraphBuilder, MoveShapeAndSizeNodesKeepsGraphInputReader) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  // Shape reads a graph input (no producing node), so it stays in place.
  const std::vector<std::string> n = builder.MakeNode("Neg", {"x"});
  const std::vector<std::string> sh = builder.MakeNode("Shape", {"x"});
  builder.MakeOutput(n[0]);
  builder.MakeOutput(sh[0]);

  EXPECT_EQ(builder.MoveShapeAndSizeNodes(), 0u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Shape");
}

TEST(GraphBuilder, MoveShapeAndSizeNodesRecursesIntoSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

  core::builder::GraphBuilder &body = builder.MakeSubgraph("body");
  body.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> p = body.MakeNode("Neg", {"a"});
  const std::vector<std::string> u = body.MakeNode("Add", {p[0], p[0]});
  const std::vector<std::string> sh = body.MakeNode("Shape", {p[0]});
  body.MakeOutput(u[0]);
  body.MakeOutput(sh[0]);

  const std::vector<std::string> top = builder.MakeNode("Neg", {"x"});
  builder.MakeOutput(top[0]);

  EXPECT_EQ(builder.MoveShapeAndSizeNodes(), 1u);
  ASSERT_EQ(builder.Subgraph("body").Nodes().size(), 3u);
  EXPECT_EQ(builder.Subgraph("body").Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[1].op_type().value(), "Shape");
  EXPECT_EQ(builder.Subgraph("body").Nodes()[2].op_type().value(), "Add");
}

TEST(GraphBuilder, InlineLocalFunctionsExpandsCallSite) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});

  // A local function computing Neg(Add(a, b)).
  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("MyFct", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  fct.MakeInput("b", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> sum = fct.MakeNode("Add", {"a", "b"});
  const std::vector<std::string> neg = fct.MakeNode("Neg", {sum[0]});
  fct.MakeOutput(neg[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("z", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> call = builder.MakeNode("MyFct", {"x", "z"}, {}, "custom");
  builder.MakeOutput(call[0]);

  EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
  // The call node is replaced by the two body nodes, and the fully inlined
  // function definition is dropped.
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Add");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Neg");
  // Formal inputs are rewired to the call inputs.
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  EXPECT_EQ(builder.Nodes()[0].input(1), "z");
  // The formal output is rewired to the call output.
  EXPECT_EQ(builder.Nodes()[1].output(0), call[0]);
  EXPECT_FALSE(builder.HasLocalFunction("MyFct"));
  // A second pass has nothing left to inline.
  EXPECT_EQ(builder.InlineLocalFunctions(), 0u);
}

TEST(GraphBuilder, InlineLocalFunctionsMovesUnchangedNodes) {
  for (bool with_call : {false, true}) {
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.SetOpsetVersion("custom", 1);
    auto &function = builder.MakeLocalFunction("Local", "custom");
    function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    function.MakeNode("Identity", {"a"}, {"result"});
    function.MakeOutput("result");

    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    std::string previous = "x";
    std::vector<const char *> unchanged_outputs;
    for (int i = 0; i < 256; ++i) {
      if (with_call && i == 128) {
        previous = builder.MakeNode("Local", {previous}, {}, "custom")[0];
      }
      const std::string output =
          "long_output_name_to_detect_node_copies_" + std::string(128, 'a') + std::to_string(i);
      builder.MakeNode("Identity", {previous}, {output});
      previous = output;
      unchanged_outputs.push_back(builder.Nodes()[builder.Nodes().size() - 1].output(0).data());
    }
    builder.MakeOutput(previous);

    EXPECT_EQ(builder.InlineLocalFunctions(), with_call ? 1u : 0u);
    ASSERT_EQ(builder.Nodes().size(), 256u + (with_call ? 1u : 0u));
    for (std::size_t i = 0; i < unchanged_outputs.size(); ++i) {
      const std::size_t index = i + (with_call && i >= 128 ? 1u : 0u);
      EXPECT_EQ(builder.Nodes()[index].output(0).data(), unchanged_outputs[i]) << i;
    }
    EXPECT_EQ(builder.BuildGraph().node().size(), builder.Nodes().size());
  }
}

TEST(GraphBuilder, InlineLocalFunctionsRewiresIntermediates) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});

  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("MyFct", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> inner = fct.MakeNode("Neg", {"a"});
  const std::vector<std::string> outer = fct.MakeNode("Neg", {inner[0]});
  fct.MakeOutput(outer[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> call = builder.MakeNode("MyFct", {"x"}, {}, "custom");
  builder.MakeOutput(call[0]);

  EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  // The intermediate value is renamed to a fresh, unused name that chains the
  // two body nodes together.
  const std::string intermediate = builder.Nodes()[0].output(0);
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  EXPECT_EQ(builder.Nodes()[1].input(0), intermediate);
  EXPECT_NE(intermediate, "x");
  EXPECT_EQ(builder.Nodes()[1].output(0), call[0]);
}

TEST(GraphBuilder, InlineLocalFunctionsExpandsNestedCalls) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});

  // Inner function Neg(a).
  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &inner = builder.MakeLocalFunction("Inner", "custom");
  inner.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> inner_out = inner.MakeNode("Neg", {"a"});
  inner.MakeOutput(inner_out[0]);

  // Outer function calls Inner and forwards its result.
  core::builder::GraphBuilder &outer = builder.MakeLocalFunction("Outer", "custom");
  outer.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> nested = outer.MakeNode("Inner", {"a"}, {}, "custom");
  outer.MakeOutput(nested[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> call = builder.MakeNode("Outer", {"x"}, {}, "custom");
  builder.MakeOutput(call[0]);

  // Both the outer call and the nested inner call are expanded.
  EXPECT_EQ(builder.InlineLocalFunctions(), 2u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "Neg");
  EXPECT_EQ(builder.Nodes()[0].input(0), "x");
  EXPECT_EQ(builder.Nodes()[0].output(0), call[0]);
  EXPECT_FALSE(builder.HasLocalFunction("Outer"));
  EXPECT_FALSE(builder.HasLocalFunction("Inner"));
}

TEST(GraphBuilder, ExportsNestedFunctionsInDependencyOrder) {
  for (const bool nested_call : {false, true}) {
    SCOPED_TRACE(nested_call);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 18}, {"custom", 1}});
    auto &base = builder.MakeLocalFunction("Base", "custom");
    base.MakeInput("value", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    base.MakeNode("Neg", {"value"}, {"result"});
    base.MakeOutput("result");

    auto &wrapper = builder.MakeSubgraph("scope").MakeLocalFunction("Wrapper", "custom");
    wrapper.MakeInput("value", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    if (nested_call) {
      wrapper.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
      utils::RepeatedProtoField<AttributeProto> attributes;
      for (const std::string name : {"then_branch", "else_branch"}) {
        auto &branch = wrapper.MakeSubgraph(name);
        branch.MakeNode("Base", {"value"}, {"Y"}, "custom");
        branch.MakeOutput("Y");
        auto &reference = attributes.add();
        reference.set_name(name + "_ref");
        reference.set_type(AttributeProto::AttributeType::STRING);
        reference.set_s(name);
      }
      wrapper.MakeNode("If", {"condition"}, {"result"}, "", "", attributes);
    } else {
      wrapper.MakeNode("Base", {"value"}, {"result"}, "custom");
    }
    wrapper.MakeOutput("result");
    const auto model = builder.ToModel();
    ASSERT_EQ(model.functions().size(), 2u);
    EXPECT_EQ(model.functions(0).name(), "Base");
    EXPECT_EQ(model.functions(1).name(), "Wrapper");
    EXPECT_NO_THROW(core::builder::GraphBuilder(model, SchemaLookup()));
  }
}

TEST(GraphBuilder, InlineLocalFunctionsClonesControlFlowSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &function = builder.MakeLocalFunction("Choose", "custom");
  function.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  function.MakeInput("value", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = function.MakeSubgraph(name);
    AttributeProto type_reference;
    type_reference.set_name("to");
    type_reference.set_ref_attr_name("dtype");
    type_reference.set_type(AttributeProto::AttributeType::INT);
    utils::RepeatedProtoField<AttributeProto> branch_attributes;
    branch_attributes.push_back(type_reference);
    branch.MakeNode("Cast", {"value"}, {"Y"}, "", "", branch_attributes);
    branch.MakeOutput("Y");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  function.MakeNode("If", {"condition"}, {"result"}, "", "", attributes);
  function.MakeOutput("result");

  builder.MakeInput("predicate", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  AttributeProto dtype;
  dtype.set_name("dtype");
  dtype.set_type(AttributeProto::AttributeType::INT);
  dtype.set_i(TensorProto::DataType::FLOAT);
  utils::RepeatedProtoField<AttributeProto> call_attributes;
  call_attributes.push_back(dtype);
  builder.MakeNode("Choose", {"predicate", "X"}, {"Y"}, "custom", "", call_attributes);
  builder.MakeOutput("Y");

  EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "If");
  EXPECT_FALSE(builder.HasLocalFunction("Choose"));
  ASSERT_EQ(builder.Subgraphs().size(), 2u);
  const GraphProto graph = builder.BuildGraph();
  ASSERT_EQ(graph.node()[0].attribute().size(), 2u);
  for (const auto &attribute : graph.node()[0].attribute()) {
    ASSERT_TRUE(attribute.has_g());
    ASSERT_EQ(attribute.g().node().size(), 1u);
    EXPECT_EQ(attribute.g().node()[0].input()[0], "X");
    EXPECT_EQ(attribute.g().node()[0].output()[0], "Y");
    ASSERT_EQ(attribute.g().node()[0].attribute().size(), 1u);
    EXPECT_TRUE(attribute.g().node()[0].attribute()[0].ref_attr_name().empty());
    EXPECT_EQ(attribute.g().node()[0].attribute()[0].i(), TensorProto::DataType::FLOAT);
  }
}

TEST(GraphBuilder, InlineLocalFunctionsExpandsCallsInClonedSubgraphs) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("custom", 1);
  auto &inner = builder.MakeLocalFunction("Inner", "custom");
  inner.MakeInput("value", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  inner.MakeNode("Identity", {"value"}, {"result"});
  inner.MakeOutput("result");

  auto &outer = builder.MakeLocalFunction("Outer", "custom");
  outer.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  outer.MakeInput("value", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = outer.MakeSubgraph(name);
    branch.MakeNode("Inner", {"value"}, {name + "_result"}, "custom");
    branch.MakeOutput(name + "_result");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  outer.MakeNode("If", {"condition"}, {"result"}, "", "", attributes);
  outer.MakeOutput("result");

  builder.MakeInput("predicate", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Outer", {"predicate", "X"}, {"Y"}, "custom");
  builder.MakeOutput("Y");

  EXPECT_EQ(builder.InlineLocalFunctions(), 3u);
  EXPECT_FALSE(builder.HasLocalFunction("Outer"));
  EXPECT_FALSE(builder.HasLocalFunction("Inner"));
  const GraphProto graph = builder.BuildGraph();
  for (const auto &attribute : graph.node()[0].attribute()) {
    ASSERT_TRUE(attribute.has_g());
    ASSERT_EQ(attribute.g().node().size(), 1u);
    EXPECT_EQ(attribute.g().node()[0].op_type(), "Identity");
  }
}

TEST(GraphBuilder, InlineLocalFunctionsMaterializesGraphCallAttributes) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("custom", 1);
  auto &inner = builder.MakeLocalFunction("Inner", "custom");
  inner.MakeInput("value", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  inner.MakeNode("Identity", {"value"}, {"result"});
  inner.MakeOutput("result");

  auto &function = builder.MakeLocalFunction("WithGraph", "custom");
  function.MakeInput("value", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  AttributeProto body_reference;
  body_reference.set_name("body");
  body_reference.set_ref_attr_name("body");
  body_reference.set_type(AttributeProto::AttributeType::GRAPH);
  NodeProto wrapper_node = MakeNode("Identity", {"value"}, {"wrapper_result"});
  wrapper_node.add_attribute(std::move(body_reference));
  AttributeProto wrapper;
  wrapper.set_name("wrapper");
  wrapper.set_type(AttributeProto::AttributeType::GRAPH);
  wrapper.mutable_g()->set_name("wrapper");
  wrapper.mutable_g()->add_node(std::move(wrapper_node));
  wrapper.mutable_g()->add_output()->set_name("wrapper_result");
  utils::RepeatedProtoField<AttributeProto> function_attributes;
  function_attributes.push_back(std::move(wrapper));
  function.MakeNode("Identity", {"value"}, {"result"}, "", "", function_attributes);
  function.MakeOutput("result");

  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("value", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  AttributeProto body;
  body.set_name("body");
  body.set_type(AttributeProto::AttributeType::GRAPH);
  body.mutable_g()->set_name("call_body");
  body.mutable_g()->add_node(MakeNode("Inner", {"value"}, {"branch_result"}, "custom"));
  body.mutable_g()->add_output()->set_name("branch_result");
  utils::RepeatedProtoField<AttributeProto> call_attributes;
  call_attributes.push_back(body);
  builder.MakeNode("WithGraph", {"X"}, {"Y"}, "custom", "", call_attributes);
  builder.MakeOutput("Y");

  EXPECT_EQ(builder.InlineLocalFunctions(), 2u);
  ASSERT_EQ(builder.Subgraphs().size(), 1u);
  const GraphProto graph = builder.BuildGraph();
  ASSERT_EQ(graph.node().size(), 1u);
  ASSERT_EQ(graph.node()[0].attribute().size(), 1u);
  const auto &inlined_wrapper = graph.node()[0].attribute()[0];
  EXPECT_EQ(inlined_wrapper.name(), "wrapper");
  ASSERT_TRUE(inlined_wrapper.has_g());
  ASSERT_EQ(inlined_wrapper.g().node().size(), 1u);
  const auto &wrapper_body = inlined_wrapper.g().node()[0];
  EXPECT_EQ(wrapper_body.input()[0], "X");
  ASSERT_EQ(wrapper_body.attribute().size(), 1u);
  ASSERT_TRUE(wrapper_body.attribute()[0].has_g());
  ASSERT_EQ(wrapper_body.attribute()[0].g().node().size(), 1u);
  EXPECT_EQ(wrapper_body.attribute()[0].g().node()[0].op_type(), "Identity");
  EXPECT_EQ(wrapper_body.attribute()[0].g().node()[0].input()[0], "value");
}

TEST(GraphBuilder, InlineLocalFunctionsPrunesOnlyUnreferencedReplacedCallSubgraphs) {
  for (bool retain_original : {false, true}) {
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    builder.SetOpsetVersion("custom", 1);
    auto &function = builder.MakeLocalFunction("F", "custom");
    function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    function.MakeNode("Identity", {"a"}, {"result"});
    function.MakeOutput("result");
    builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));

    auto &original = builder.MakeSubgraph("original");
    original.MakeNode("Identity", {"X"}, {"branch_result"});
    original.MakeOutput("branch_result");
    AttributeProto reference;
    reference.set_name("body_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s("original");
    utils::RepeatedProtoField<AttributeProto> attributes;
    attributes.push_back(std::move(reference));
    builder.MakeNode("F", {"X"}, {"Y"}, "custom", "", attributes);
    if (retain_original) {
      builder.MakeNode("Identity", {"Y"}, {"Z"}, "", "", attributes);
    }
    builder.MakeOutput(retain_original ? "Z" : "Y");
    auto &pending = builder.MakeSubgraph("pending");
    pending.MakeSubgraph("nested_pending");

    EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
    EXPECT_EQ(builder.HasSubgraph("original"), retain_original);
    ASSERT_TRUE(builder.HasSubgraph("pending"));
    EXPECT_EQ(&builder.Subgraph("pending"), &pending);
    EXPECT_TRUE(pending.HasSubgraph("nested_pending"));
    EXPECT_EQ(builder.Subgraphs().size(), retain_original ? 2u : 1u);
    pending.MakeNode("Identity", {"X"}, {"pending_result"});
    pending.MakeOutput("pending_result");
    AttributeProto pending_reference;
    pending_reference.set_name("body_ref");
    pending_reference.set_type(AttributeProto::AttributeType::STRING);
    pending_reference.set_s("pending");
    utils::RepeatedProtoField<AttributeProto> pending_attributes;
    pending_attributes.push_back(std::move(pending_reference));
    builder.MakeNode("Identity", {"Y"}, {"attached"}, "", "", pending_attributes);
    const GraphProto graph = builder.BuildGraph();
    ASSERT_FALSE(graph.node().empty());
    const auto &attached_attributes = graph.node()[graph.node().size() - 1].attribute();
    ASSERT_EQ(attached_attributes.size(), 1u);
    ASSERT_TRUE(attached_attributes[0].has_g());
    ASSERT_EQ(attached_attributes[0].g().output().size(), 1u);
    EXPECT_EQ(attached_attributes[0].g().output()[0].name(), "pending_result");
  }
}

TEST(GraphBuilder, InlineLocalFunctionsPreservesSubgraphSparseInitializerNames) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset("", 23);
  model.add_opset("custom", 1);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("g");
  auto *predicate = graph->add_input();
  predicate->set_name("predicate");
  predicate->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::BOOL);
  auto *value = graph->add_input();
  value->set_name("X");
  value->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::FLOAT);
  graph->add_initializer(MakeInitializer<float>("replacement", {2}, {2.0f, 3.0f}));
  graph->add_initializer(MakeInitializer<float>("Inner_tmp_0", {2}, {2.0f, 3.0f}));
  graph->add_node(MakeNode("Choose", {"predicate", "X"}, {"Y"}, "custom"));
  graph->add_output()->set_name("Y");

  FunctionProto inner;
  inner.set_name("Inner");
  inner.set_domain("custom");
  inner.add_input("a");
  inner.add_output("r");
  inner.add_opset("", 23);
  inner.add_node(MakeNode("Identity", {"a"}, {"tmp"}));
  inner.add_node(MakeNode("Identity", {"tmp"}, {"r"}));
  model.add_function(std::move(inner));

  FunctionProto function;
  function.set_name("Choose");
  function.set_domain("custom");
  function.add_input("condition");
  function.add_input("value");
  function.add_output("result");
  function.add_opset("", 23);
  function.add_opset("custom", 1);
  NodeProto conditional = MakeNode("If", {"condition"}, {"result"});
  for (const std::string name : {"then_branch", "else_branch"}) {
    AttributeProto branch;
    branch.set_name(name);
    branch.set_type(AttributeProto::AttributeType::GRAPH);
    branch.mutable_g()->set_name(name);
    auto *sparse = branch.mutable_g()->add_sparse_initializer();
    sparse->add_dims(1);
    sparse->mutable_values()->set_name("value");
    sparse->mutable_values()->set_data_type(TensorProto::DataType::FLOAT);
    sparse->mutable_values()->add_dims(1);
    sparse->mutable_values()->add_float_data(1.0f);
    sparse->mutable_indices()->set_data_type(TensorProto::DataType::INT64);
    sparse->mutable_indices()->add_dims(1);
    sparse->mutable_indices()->add_int64_data(0);
    auto *collision = branch.mutable_g()->add_sparse_initializer();
    collision->add_dims(1);
    collision->mutable_values()->set_data_type(TensorProto::DataType::FLOAT);
    collision->mutable_values()->add_dims(1);
    collision->mutable_values()->add_float_data(1.0f);
    collision->mutable_indices()->set_data_type(TensorProto::DataType::INT64);
    collision->mutable_indices()->add_dims(1);
    collision->mutable_indices()->add_int64_data(0);
    collision->mutable_values()->set_name("Inner_tmp_0");
    branch.mutable_g()->add_node(MakeNode("Inner", {"Inner_tmp_0"}, {"inner_result"}, "custom"));
    branch.mutable_g()->add_output()->set_name("value");
    NodeProto nested = MakeNode("Identity", {"condition"}, {"ignored"});
    AttributeProto nested_body;
    nested_body.set_name("body");
    nested_body.set_type(AttributeProto::AttributeType::GRAPH);
    nested_body.mutable_g()->set_name("nested");
    nested_body.mutable_g()->add_node(MakeNode("Identity", {"value"}, {"nested_value"}));
    nested_body.mutable_g()->add_output()->set_name("nested_value");
    nested.add_attribute(std::move(nested_body));
    branch.mutable_g()->add_node(std::move(nested));
    auto *annotation = branch.mutable_g()->add_quantization_annotation();
    annotation->set_tensor_name("condition");
    auto *parameter = annotation->add_quant_parameter_tensor_names();
    parameter->set_key("SCALE_TENSOR");
    parameter->set_value("value");
    conditional.add_attribute(std::move(branch));
  }
  function.add_node(std::move(conditional));
  model.add_function(std::move(function));

  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_EQ(builder.InlineLocalFunctions(), 3u);
  for (const auto &subgraph : builder.Subgraphs()) {
    EXPECT_FALSE(subgraph->HasShape("Inner_tmp_0"));
  }
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 1u);
  const GraphProto inlined = builder.ToGraph();
  for (const auto &attribute : inlined.node()[0].attribute()) {
    ASSERT_TRUE(attribute.has_g());
    ASSERT_EQ(attribute.g().sparse_initializer().size(), 2u);
    EXPECT_EQ(attribute.g().sparse_initializer()[0].values().name(), "value");
    EXPECT_EQ(attribute.g().sparse_initializer()[1].values().name(), "Inner_tmp_0");
    ASSERT_EQ(attribute.g().output().size(), 1u);
    EXPECT_EQ(attribute.g().output()[0].name(), "value");
    ASSERT_EQ(attribute.g().node().size(), 3u);
    EXPECT_EQ(attribute.g().node()[0].input()[0], "Inner_tmp_0");
    EXPECT_EQ(attribute.g().node()[0].output()[0], "Inner_tmp_1");
    EXPECT_EQ(attribute.g().node()[1].output()[0], "inner_result");
    ASSERT_EQ(attribute.g().node()[2].attribute().size(), 1u);
    ASSERT_TRUE(attribute.g().node()[2].attribute()[0].has_g());
    ASSERT_EQ(attribute.g().node()[2].attribute()[0].g().output().size(), 1u);
    EXPECT_EQ(attribute.g().node()[2].attribute()[0].g().node()[0].input()[0], "value");
    EXPECT_EQ(attribute.g().node()[2].attribute()[0].g().output()[0].name(), "nested_value");
    ASSERT_EQ(attribute.g().quantization_annotation().size(), 1u);
    const auto &annotation = attribute.g().quantization_annotation()[0];
    EXPECT_EQ(annotation.tensor_name(), "predicate");
    ASSERT_EQ(annotation.quant_parameter_tensor_names().size(), 1u);
    EXPECT_EQ(annotation.quant_parameter_tensor_names()[0].value(), "value");
  }
}

TEST(GraphBuilder, InlineLocalFunctionsCleansCallsFromGraphDefaults) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset("", 23);
  model.add_opset("custom", 1);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("g");
  auto *input = graph->add_input();
  input->set_name("X");
  input->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::FLOAT);
  graph->add_node(MakeNode("Outer", {"X"}, {"Y"}, "custom"));
  graph->add_output()->set_name("Y");

  FunctionProto inner;
  inner.set_name("Inner");
  inner.set_domain("custom");
  inner.add_input("value");
  inner.add_output("result");
  inner.add_opset("", 23);
  inner.add_opset("custom", 1);
  inner.add_node(MakeNode("Identity", {"value"}, {"result"}));
  model.add_function(inner);

  FunctionProto outer;
  outer.set_name("Outer");
  outer.set_domain("custom");
  outer.add_input("value");
  outer.add_output("result");
  outer.add_opset("", 23);
  outer.add_opset("custom", 1);
  outer.add_attribute("body");
  auto *default_body = outer.add_attribute_proto();
  default_body->set_name("body");
  default_body->set_type(AttributeProto::AttributeType::GRAPH);
  default_body->mutable_g()->set_name("default_body");
  default_body->mutable_g()->add_node(MakeNode("Inner", {"value"}, {"nested"}, "custom"));
  default_body->mutable_g()->add_output()->set_name("nested");
  NodeProto body = MakeNode("Identity", {"value"}, {"result"});
  AttributeProto body_reference;
  body_reference.set_name("body");
  body_reference.set_ref_attr_name("body");
  body_reference.set_type(AttributeProto::AttributeType::GRAPH);
  body.add_attribute(body_reference);
  outer.add_node(body);
  model.add_function(outer);

  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_EQ(builder.InlineLocalFunctions(), 2u);
  EXPECT_FALSE(builder.HasLocalFunction("Outer"));
  EXPECT_FALSE(builder.HasLocalFunction("Inner"));
  EXPECT_TRUE(builder.ToModel().functions().empty());
}

TEST(GraphBuilder, InlineLocalFunctionsRetainsDefinitionsCalledByExportedDefaults) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset("", 23);
  model.add_opset("custom", 1);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("g");
  graph->add_input()->set_name("X");
  graph->add_node(MakeNode("Inner", {"X"}, {"Y"}, "custom"));
  graph->add_output()->set_name("Y");

  FunctionProto inner;
  inner.set_name("Inner");
  inner.set_domain("custom");
  inner.add_input("value");
  inner.add_output("result");
  inner.add_opset("", 23);
  inner.add_opset("custom", 1);
  inner.add_node(MakeNode("Identity", {"value"}, {"result"}));
  model.add_function(inner);

  FunctionProto exported;
  exported.set_name("Exported");
  exported.set_domain("custom");
  exported.add_input("value");
  exported.add_output("result");
  exported.add_opset("", 23);
  exported.add_opset("custom", 1);
  exported.add_attribute("body");
  auto *default_body = exported.add_attribute_proto();
  default_body->set_name("body");
  default_body->set_type(AttributeProto::AttributeType::GRAPH);
  default_body->mutable_g()->set_name("default_body");
  default_body->mutable_g()->add_node(MakeNode("Inner", {"value"}, {"nested"}, "custom"));
  default_body->mutable_g()->add_output()->set_name("nested");
  NodeProto body = MakeNode("Identity", {"value"}, {"result"});
  AttributeProto body_reference;
  body_reference.set_name("body");
  body_reference.set_ref_attr_name("body");
  body_reference.set_type(AttributeProto::AttributeType::GRAPH);
  body.add_attribute(body_reference);
  exported.add_node(body);
  model.add_function(exported);

  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
  EXPECT_TRUE(builder.HasLocalFunction("Inner"));
  EXPECT_TRUE(builder.HasLocalFunction("Exported"));
  EXPECT_EQ(builder.ToModel().functions().size(), 2u);
}

TEST(GraphBuilder, InlineLocalFunctionsKeepsUncalledFunction) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});

  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("MyFct", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> out = fct.MakeNode("Neg", {"a"});
  fct.MakeOutput(out[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> top = builder.MakeNode("Neg", {"x"});
  builder.MakeOutput(top[0]);

  // No call site, so nothing is inlined and the definition is left in place.
  EXPECT_EQ(builder.InlineLocalFunctions(), 0u);
  EXPECT_TRUE(builder.HasLocalFunction("MyFct"));
}

TEST(GraphBuilder, InlineLocalFunctionsPreservesAndMatchesOverloads) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset("", 23);
  model.add_opset("custom", 1);
  GraphProto *graph = model.mutable_graph();
  graph->set_name("g");
  graph->add_input()->set_name("X");
  graph->add_node(MakeNode("Outer", {"X"}, {"Y"}, "custom"));
  graph->add_node(MakeNode("Inner", {"X"}, {"Z"}, "custom"));
  graph->add_output()->set_name("Y");
  graph->add_output()->set_name("Z");

  FunctionProto inner;
  inner.set_name("Inner");
  inner.set_domain("custom");
  inner.set_overload("typed");
  inner.add_input("a");
  inner.add_output("r");
  inner.add_opset("", 23);
  inner.add_node(MakeNode("Identity", {"a"}, {"r"}));
  model.add_function(std::move(inner));

  FunctionProto outer;
  outer.set_name("Outer");
  outer.set_domain("custom");
  outer.add_input("a");
  outer.add_output("r");
  outer.add_opset("", 23);
  NodeProto inner_call = MakeNode("Inner", {"a"}, {"r"}, "custom");
  inner_call.set_overload("typed");
  inner_call.set_doc_string("preserved documentation");
  inner_call.add_metadata("purpose", "preserve-node-fields");
  outer.add_node(std::move(inner_call));
  model.add_function(std::move(outer));

  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_EQ(builder.InlineLocalFunctions({{"custom", "Outer"}}), 1u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type(), "Inner");
  EXPECT_EQ(builder.Nodes()[0].overload(), "typed");
  EXPECT_EQ(builder.Nodes()[0].doc_string(), "preserved documentation");
  ASSERT_EQ(builder.Nodes()[0].metadata_props().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].metadata_props()[0].value(), "preserve-node-fields");
  EXPECT_TRUE(builder.Nodes()[1].overload().empty());

  EXPECT_EQ(builder.InlineLocalFunctions({{"custom", "Inner"}}), 1u);
  ASSERT_EQ(builder.Nodes().size(), 2u);
  EXPECT_EQ(builder.Nodes()[0].op_type(), "Identity");
  EXPECT_EQ(builder.Nodes()[1].op_type(), "Inner");
  EXPECT_TRUE(builder.Nodes()[1].overload().empty());
  EXPECT_FALSE(builder.HasLocalFunction("Inner"));
}

TEST(GraphBuilder, InlineLocalFunctionsResolvesAttributeReferences) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});

  // A local function whose body forwards a function attribute to a body node.
  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("Scaled", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  AttributeProto ref;
  ref.set_name("alpha");
  ref.set_ref_attr_name("alpha");
  ref.set_type(AttributeProto::AttributeType::FLOAT);
  utils::RepeatedProtoField<AttributeProto> body_attrs;
  body_attrs.push_back(ref);
  const std::vector<std::string> scaled = fct.MakeNode("LeakyRelu", {"a"}, {}, "", "", body_attrs);
  fct.MakeOutput(scaled[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  AttributeProto alpha;
  alpha.set_name("alpha");
  alpha.set_ref_attr_name("dangling");
  alpha.set_type(AttributeProto::AttributeType::FLOAT);
  alpha.set_f(0.25f);
  utils::RepeatedProtoField<AttributeProto> call_attrs;
  call_attrs.push_back(alpha);
  const std::vector<std::string> call =
      builder.MakeNode("Scaled", {"x"}, {}, "custom", "", call_attrs);
  builder.MakeOutput(call[0]);

  EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
  ASSERT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "LeakyRelu");
  ASSERT_EQ(builder.Nodes()[0].attribute().size(), 1);
  const AttributeProto &resolved = builder.Nodes()[0].attribute()[0];
  EXPECT_EQ(resolved.name().value(), "alpha");
  EXPECT_TRUE(resolved.ref_attr_name().empty());
  EXPECT_FLOAT_EQ(resolved.f(), 0.25f);
}

TEST(GraphBuilder, LocalFunctionIdentityNormalizesAllDefaultDomainAliases) {
  for (const std::string domain : {"", "ai.onnx", "onnx.ai"}) {
    SCOPED_TRACE(domain);
    core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
    FunctionProto definition;
    definition.set_name("Alias");
    definition.set_domain(domain);
    definition.set_overload("typed");
    definition.add_opset("", 28);
    definition.add_input("X");
    definition.add_output("Y");
    definition.add_node("Neg", {"X"}, {"Y"});
    auto &function = builder.MakeLocalFunction(definition);
    auto &branch = function.MakeSubgraph("branch");
    auto &nested = branch.MakeSubgraph("nested");
    for (const std::string query_domain : {"", "ai.onnx", "onnx.ai"}) {
      EXPECT_TRUE(builder.Shapes().HasLocalFunction(
          core::shapes::ShapesContext::LocalFunctionKey(query_domain, "Alias", "typed")));
      EXPECT_TRUE(function.IsLocalFunction("Alias", query_domain, "typed"));
      EXPECT_TRUE(nested.IsInsideLocalFunction("Alias", query_domain, "typed"));
      EXPECT_FALSE(function.IsLocalFunction("Alias", query_domain));
      EXPECT_FALSE(nested.IsInsideLocalFunction("Alias", query_domain, "other"));
      EXPECT_FALSE(branch.IsLocalFunction("Alias", query_domain, "typed"));
    }
    EXPECT_FALSE(function.IsLocalFunction("Alias", "custom", "typed"));
    EXPECT_FALSE(nested.IsInsideLocalFunction("Other", domain, "typed"));
    EXPECT_FALSE(builder.IsInsideLocalFunction("Alias", domain, "typed"));
  }
}

TEST(GraphBuilder, InlineLocalFunctionsRejectsMissingAttributeReferences) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});

  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &fct = builder.MakeLocalFunction("Scaled", "custom");
  fct.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  fct.MakeInitializer(MakeInitializer<float>("weight", {1}, {1.0f}));
  fct.MakeNode("Identity", {"a"}, {"weighted"});
  auto &branch = fct.MakeSubgraph("body");
  branch.MakeNode("Identity", {"a"}, {"branch_result"});
  branch.MakeOutput("branch_result");
  AttributeProto graph_ref;
  graph_ref.set_name("body_ref");
  graph_ref.set_type(AttributeProto::AttributeType::STRING);
  graph_ref.set_s("body");
  utils::RepeatedProtoField<AttributeProto> graph_attrs;
  graph_attrs.push_back(std::move(graph_ref));
  fct.MakeNode("Identity", {"weighted"}, {"intermediate"}, "", "", graph_attrs);
  AttributeProto ref;
  ref.set_name("alpha");
  ref.set_ref_attr_name("alpha");
  ref.set_type(AttributeProto::AttributeType::FLOAT);
  utils::RepeatedProtoField<AttributeProto> body_attrs;
  body_attrs.push_back(ref);
  const std::vector<std::string> scaled =
      fct.MakeNode("LeakyRelu", {"intermediate"}, {}, "", "", body_attrs);
  fct.MakeOutput(scaled[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("Neg", {"x"}, {"negated"});
  const std::vector<std::string> call = builder.MakeNode("Scaled", {"negated"}, {}, "custom");
  builder.MakeOutput(call[0]);

  EXPECT_THROW(builder.InlineLocalFunctions(), core::builder::BuilderError);
}

TEST(GraphBuilder, InlineLocalFunctionsIncludeSelectsFunctions) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});

  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &keep = builder.MakeLocalFunction("Keep", "custom");
  keep.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_out = keep.MakeNode("Neg", {"a"});
  keep.MakeOutput(keep_out[0]);

  core::builder::GraphBuilder &drop = builder.MakeLocalFunction("Drop", "custom");
  drop.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> drop_out = drop.MakeNode("Neg", {"a"});
  drop.MakeOutput(drop_out[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_call = builder.MakeNode("Keep", {"x"}, {}, "custom");
  const std::vector<std::string> drop_call = builder.MakeNode("Drop", {keep_call[0]}, {}, "custom");
  builder.MakeOutput(drop_call[0]);

  // Only "Drop" is inlined; the "Keep" call and its definition are untouched.
  EXPECT_EQ(builder.InlineLocalFunctions({{"custom", "Drop"}}), 1u);
  EXPECT_TRUE(builder.HasLocalFunction("Keep"));
  EXPECT_FALSE(builder.HasLocalFunction("Drop"));
  bool has_keep_call = false;
  for (const auto &node : builder.Nodes()) {
    if (node.op_type().value() == "Keep") {
      has_keep_call = true;
    }
  }
  EXPECT_TRUE(has_keep_call);
}

TEST(GraphBuilder, InlineLocalFunctionsExcludeSkipsFunctions) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});

  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &keep = builder.MakeLocalFunction("Keep", "custom");
  keep.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_out = keep.MakeNode("Neg", {"a"});
  keep.MakeOutput(keep_out[0]);

  core::builder::GraphBuilder &drop = builder.MakeLocalFunction("Drop", "custom");
  drop.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> drop_out = drop.MakeNode("Neg", {"a"});
  drop.MakeOutput(drop_out[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_call = builder.MakeNode("Keep", {"x"}, {}, "custom");
  const std::vector<std::string> drop_call = builder.MakeNode("Drop", {keep_call[0]}, {}, "custom");
  builder.MakeOutput(drop_call[0]);

  // Everything except "Keep" is inlined.
  EXPECT_EQ(builder.InlineLocalFunctions({}, {{"custom", "Keep"}}), 1u);
  EXPECT_TRUE(builder.HasLocalFunction("Keep"));
  EXPECT_FALSE(builder.HasLocalFunction("Drop"));
}

TEST(GraphBuilder, InlineLocalFunctionsIncludeWildcardsDomainByName) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});

  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &keep = builder.MakeLocalFunction("Keep", "custom");
  keep.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_out = keep.MakeNode("Neg", {"a"});
  keep.MakeOutput(keep_out[0]);

  core::builder::GraphBuilder &drop = builder.MakeLocalFunction("Drop", "custom");
  drop.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> drop_out = drop.MakeNode("Neg", {"a"});
  drop.MakeOutput(drop_out[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> keep_call = builder.MakeNode("Keep", {"x"}, {}, "custom");
  const std::vector<std::string> drop_call = builder.MakeNode("Drop", {keep_call[0]}, {}, "custom");
  builder.MakeOutput(drop_call[0]);

  // An empty domain matches the function by name regardless of its domain.
  EXPECT_EQ(builder.InlineLocalFunctions({{"", "Drop"}}), 1u);
  EXPECT_TRUE(builder.HasLocalFunction("Keep"));
  EXPECT_FALSE(builder.HasLocalFunction("Drop"));
}

TEST(GraphBuilder, InlineLocalFunctionsIncludeWildcardsNameByDomain) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});

  builder.SetOpsetVersion("custom", 1);
  core::builder::GraphBuilder &fa = builder.MakeLocalFunction("First", "custom");
  fa.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> fa_out = fa.MakeNode("Neg", {"a"});
  fa.MakeOutput(fa_out[0]);

  builder.SetOpsetVersion("other", 1);
  core::builder::GraphBuilder &fb = builder.MakeLocalFunction("Second", "other");
  fb.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> fb_out = fb.MakeNode("Neg", {"a"});
  fb.MakeOutput(fb_out[0]);

  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  const std::vector<std::string> ca = builder.MakeNode("First", {"x"}, {}, "custom");
  const std::vector<std::string> cb = builder.MakeNode("Second", {ca[0]}, {}, "other");
  builder.MakeOutput(cb[0]);

  // An empty name matches every function in the given domain only.
  EXPECT_EQ(builder.InlineLocalFunctions({{"custom", ""}}), 1u);
  EXPECT_FALSE(builder.HasLocalFunction("First"));
  EXPECT_TRUE(builder.HasLocalFunction("Second"));
}

TEST(GraphBuilder, InlineLocalFunctionsRejectsIncludeAndExclude) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  EXPECT_THROW(builder.InlineLocalFunctions({{"", "A"}}, {{"", "B"}}), core::builder::BuilderError);
}

TEST(GraphBuilder, NativeLocalFunctionInfersBeforeFollowingNode) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.SetOpsetVersion("local", 1);
  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("F", {"X"}, {"result"}, "local");
  ASSERT_TRUE(builder.HasShape("result"));
  EXPECT_EQ(builder.GetShape("result").Shape(), MakeShape({2, 3}));
  builder.MakeNode("Add", {"result", "X"}, {"Y"});
  builder.MakeOutput("Y");
  const ModelProto model = builder.ToModel();
  ASSERT_EQ(model.functions().size(), 1u);
  ASSERT_GE(model.functions(0).value_info().size(), 2u);
  EXPECT_EQ(model.functions(0).value_info(0).name(), "a");
  EXPECT_TRUE(model.functions(0).value_info(0).has_type());
  EXPECT_EQ(model.graph().output(0).type().tensor_type().shape().dim_size(), 2);
  core::builder::GraphBuilder imported(model, SchemaLookup());
  EXPECT_EQ(imported.LocalFunction("F").GetShape("a").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(imported.GetShape("Y").Shape(), MakeShape({2, 3}));
}

TEST(GraphBuilder, NativeLocalFunctionDefinitionsSurviveMovesAndEdits) {
  core::builder::GraphBuilder original("g", SchemaLookup(), {{"", 28}});
  original.SetOpsetVersion("local", 1);
  auto &function = original.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeNode("Shape", {"a"}, {"shape"});
  function.MakeOutput("r");
  function.MakeOutput("shape");
  original.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  original.MakeNode("F", {"X"}, {"first", "first_shape"}, "local");
  core::shapes::ShapesContext retained;
  for (const auto &[domain, version] : original.Shapes().Opsets()) {
    retained.SetOpsetVersion(domain, version);
  }
  retained.CopyLocalFunctions(original.Shapes());
  retained.CopyValueFrom("X", original.Shapes(), "X");
  core::builder::GraphBuilder moved(std::move(original));
  moved.LocalFunction("F").MakeNode("Neg", {"a"}, {"dead"});
  EXPECT_EQ(moved.LocalFunction("F").RemoveUnusedNodes(), 1u);
  const auto outputs = moved.MakeNode("F", {"X"}, {}, "local");
  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_EQ(moved.GetShape(outputs[0]).Shape(), MakeShape({2, 3}));
  EXPECT_EQ(moved.GetShape(outputs[1]).Shape(), MakeShape({2}));
  // The selectively retained context still owns the definition captured before the move.
  retained.ComputeShapeNode(MakeNode("F", {"X"}, {"old", "old_shape"}, "local"));
  EXPECT_EQ(retained.Get("old").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(retained.Get("old_shape").Shape(), MakeShape({2}));
}

TEST(GraphBuilder, NativeGenericFunctionImportsAndSpecializesNestedMatMul) {
  core::builder::GraphBuilder seed("g", SchemaLookup());
  seed.SetOpsetVersion("", 18);
  seed.SetOpsetVersion("local", 1);
  seed.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  seed.MakeInput("W", core::symbolic::TensorType::kFloat, MakeShape({3, 4}));
  ModelProto model = seed.ToModel();
  FunctionProto outer;
  outer.set_name("Outer");
  outer.set_domain("local");
  outer.add_input("a");
  outer.add_input("b");
  outer.add_output("r");
  outer.add_opset("", 18);
  outer.add_opset("local", 1);
  outer.add_node(MakeNode("Inner", {"a", "b"}, {"r"}, "local"));
  FunctionProto inner = outer;
  inner.set_name("Inner");
  inner.ref_node().clear();
  inner.add_node(MakeNode("MatMul", {"a", "b"}, {"r"}));
  model.add_function(inner);
  model.add_function(outer);
  model.mutable_graph()->add_node(MakeNode("Outer", {"X", "W"}, {"Y"}, "local"));
  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_FALSE(builder.LocalFunction("Inner").HasShape("a"));
  ASSERT_TRUE(builder.HasShape("Y"));
  EXPECT_EQ(builder.GetShape("Y").Shape(), MakeShape({2, 4}));
  builder.MakeNode("Add", {"Y", "Y"}, {"Z"});
  EXPECT_EQ(builder.GetShape("Z").Shape(), MakeShape({2, 4}));
  for (auto &function : model.ref_functions()) {
    for (std::size_t i = 0; i < model.graph().input().size(); ++i) {
      ValueInfoProto input = model.graph().input(i);
      input.set_name(function.input(i));
      function.add_value_info(input);
    }
  }
  core::builder::GraphBuilder typed(model, SchemaLookup());
  ASSERT_TRUE(typed.LocalFunction("Inner").HasShape("r"));
  EXPECT_EQ(typed.LocalFunction("Inner").GetShape("r").Shape(), MakeShape({2, 4}));
  EXPECT_EQ(typed.GetShape("Y").Shape(), MakeShape({2, 4}));
}

TEST(GraphBuilder, RejectsForwardLocalFunctionInsteadOfNativeFallback) {
  for (const std::string domain : {"", "ai.onnx", "local"}) {
    for (const bool callee_first : {false, true}) {
      SCOPED_TRACE(domain);
      SCOPED_TRACE(callee_first);
      core::builder::GraphBuilder seed("g", SchemaLookup(), {{"", 18}, {"local", 1}});
      seed.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
      ModelProto model = seed.ToModel();
      FunctionProto callee;
      callee.set_name("Neg");
      callee.set_domain(domain);
      callee.set_overload("custom");
      callee.add_input("a");
      callee.add_output("r");
      callee.add_opset("", 18);
      ValueInfoProto input = model.graph().input(0);
      input.set_name("a");
      callee.add_value_info(input);
      callee.add_node(MakeNode("Identity", {"a"}, {"r"}));
      FunctionProto wrapper = callee;
      wrapper.set_name("Wrapper");
      wrapper.set_domain("local");
      wrapper.set_overload("");
      wrapper.add_opset("local", 1);
      wrapper.ref_node().clear();
      NodeProto call = MakeNode("Neg", {"a"}, {"r"}, domain == "ai.onnx" ? "" : domain.c_str());
      call.set_overload("custom");
      wrapper.add_node(call);
      FunctionProto other_overload = callee;
      other_overload.set_overload("other");
      model.add_function(other_overload);
      model.add_function(callee_first ? callee : wrapper);
      model.add_function(callee_first ? wrapper : callee);
      model.mutable_graph()->add_node(MakeNode("Wrapper", {"X"}, {"Y"}, "local"));
      if (callee_first) {
        core::builder::GraphBuilder imported(model, SchemaLookup());
        EXPECT_EQ(imported.GetShape("Y").Shape(), MakeShape({2, 3}));
      } else {
        try {
          core::builder::GraphBuilder imported(model, SchemaLookup());
          FAIL() << "Expected an explicit forward-call error";
        } catch (const core::builder::BuilderError &error) {
          EXPECT_NE(std::string(error.what()).find("not yet imported"), std::string::npos);
          EXPECT_NE(std::string(error.what()).find("Neg"), std::string::npos);
          EXPECT_NE(std::string(error.what()).find("custom"), std::string::npos);
        }
      }
    }
  }
}

TEST(GraphBuilder, ForwardLocalFunctionCheckUsesCompleteIdentity) {
  for (const int different_field : {0, 1, 2}) {
    SCOPED_TRACE(different_field);
    core::builder::GraphBuilder seed("g", SchemaLookup(), {{"", 18}, {"local", 1}});
    seed.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    ModelProto model = seed.ToModel();
    FunctionProto wrapper;
    wrapper.set_name("Wrapper");
    wrapper.set_domain("local");
    wrapper.add_input("a");
    wrapper.add_output("r");
    wrapper.add_opset("", 18);
    NodeProto call = MakeNode("Neg", {"a"}, {"r"});
    call.set_overload("custom");
    wrapper.add_node(call);
    model.add_function(wrapper);
    FunctionProto callee = wrapper;
    callee.set_domain(different_field == 0 ? "local" : "");
    callee.set_name(different_field == 1 ? "Abs" : "Neg");
    callee.set_overload(different_field == 2 ? "other" : "custom");
    callee.ref_node().clear();
    callee.add_node(MakeNode("Identity", {"a"}, {"r"}));
    model.add_function(callee);
    model.mutable_graph()->add_node(MakeNode("Wrapper", {"X"}, {"Y"}, "local"));
    core::builder::GraphBuilder imported(model, SchemaLookup());
    EXPECT_EQ(imported.GetShape("Y").Shape(), MakeShape({2, 3}));
  }
}

TEST(GraphBuilder, RejectsForwardLocalFunctionInsideNestedGraphAttributes) {
  for (const bool repeated_graphs : {false, true}) {
    SCOPED_TRACE(repeated_graphs);
    ModelProto model;
    model.mutable_graph()->set_name("g");
    model.add_opset("", 18);
    FunctionProto wrapper;
    wrapper.set_name("Wrapper");
    wrapper.set_domain("local");
    wrapper.add_input("a");
    wrapper.add_input("condition");
    wrapper.add_output("r");
    wrapper.add_opset("", 18);
    NodeProto call = MakeNode("Neg", {"a"}, {"r"});
    call.set_overload("custom");
    for (int depth = 0; depth < 2; ++depth) {
      AttributeProto attribute;
      attribute.set_name(repeated_graphs ? "body" : "then_branch");
      attribute.set_type(repeated_graphs ? AttributeProto::GRAPHS : AttributeProto::GRAPH);
      if (repeated_graphs) {
        GraphProto *first = attribute.add_graphs();
        first->set_name("first");
        first->add_node(MakeNode("Identity", {"a"}, {"first_result"}));
        first->add_output()->set_name("first_result");
      }
      GraphProto *graph = repeated_graphs ? attribute.add_graphs() : attribute.mutable_g();
      graph->set_name("nested");
      graph->add_node(call);
      graph->add_output()->set_name("r");
      call = repeated_graphs ? MakeNode("Identity", {"a"}, {"r"})
                             : MakeNode("If", {"condition"}, {"r"});
      call.add_attribute(attribute);
      if (!repeated_graphs) {
        attribute.set_name("else_branch");
        call.add_attribute(attribute);
      }
    }
    wrapper.add_node(call);
    model.add_function(wrapper);
    FunctionProto callee = wrapper;
    callee.set_name("Neg");
    callee.set_domain("");
    callee.set_overload("custom");
    callee.ref_input().clear();
    callee.add_input("a");
    callee.ref_node().clear();
    callee.add_node(MakeNode("Identity", {"a"}, {"r"}));
    model.add_function(callee);
    try {
      core::builder::GraphBuilder imported(model, SchemaLookup());
      FAIL() << "Expected an explicit nested forward-call error";
    } catch (const core::builder::BuilderError &error) {
      EXPECT_NE(std::string(error.what()).find("not yet imported"), std::string::npos);
    }
    if (!repeated_graphs) {
      model.ref_functions().clear();
      model.add_function(callee);
      model.add_function(wrapper);
      EXPECT_NO_THROW(core::builder::GraphBuilder imported(model, SchemaLookup()));
    }
  }
}

TEST(GraphBuilder, RejectsForwardLocalFunctionInsideDefaultGraphAttributes) {
  for (const bool repeated_graphs : {false, true}) {
    SCOPED_TRACE(repeated_graphs);
    ModelProto model;
    model.mutable_graph()->set_name("g");
    model.add_opset("", 18);
    FunctionProto wrapper;
    wrapper.set_name("Wrapper");
    wrapper.set_domain("local");
    wrapper.add_input("a");
    wrapper.add_output("r");
    wrapper.add_opset("", 18);
    wrapper.add_node(MakeNode("Identity", {"a"}, {"r"}));
    AttributeProto attribute;
    attribute.set_name("body");
    attribute.set_type(repeated_graphs ? AttributeProto::GRAPHS : AttributeProto::GRAPH);
    GraphProto *graph = repeated_graphs ? attribute.add_graphs() : attribute.mutable_g();
    graph->set_name("default_body");
    NodeProto call = MakeNode("Neg", {"a"}, {"branch_result"});
    call.set_overload("custom");
    graph->add_node(call);
    graph->add_output()->set_name("branch_result");
    wrapper.add_attribute_proto(attribute);
    model.add_function(wrapper);
    FunctionProto callee = wrapper;
    callee.set_name("Neg");
    callee.set_domain("");
    callee.set_overload("custom");
    callee.ref_attribute_proto().clear();
    model.add_function(callee);
    EXPECT_THROW(core::builder::GraphBuilder imported(model, SchemaLookup()),
                 core::builder::BuilderError);
    model.ref_functions().clear();
    model.add_function(callee);
    model.add_function(wrapper);
    EXPECT_NO_THROW(core::builder::GraphBuilder imported(model, SchemaLookup()));
  }
}

TEST(GraphBuilder, NativeTypedFunctionInfersOnImportAndPreservesDeclarationsAfterMove) {
  core::builder::GraphBuilder seed("g", SchemaLookup());
  seed.SetOpsetVersion("", 18);
  seed.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  ModelProto model = seed.ToModel();
  FunctionProto function;
  function.set_name("F");
  function.set_domain("local");
  function.set_doc_string("Function documentation");
  function.set_overload("typed");
  function.add_metadata("purpose", "roundtrip");
  function.add_input("a");
  function.add_output("r");
  function.add_opset("", 18);
  function.add_attribute("required");
  AttributeProto default_attribute;
  default_attribute.set_name("optional");
  default_attribute.set_type(AttributeProto::AttributeType::INT);
  default_attribute.set_i(7);
  function.add_attribute_proto(default_attribute);
  for (const std::string name : {"a", "middle", "r"}) {
    ValueInfoProto value_info = model.graph().input(0);
    value_info.set_name(name);
    value_info.set_doc_string(name + " documentation");
    auto *metadata = value_info.add_metadata_props();
    metadata->set_key("purpose");
    metadata->set_value(name);
    function.add_value_info(value_info);
  }
  function.add_node(MakeNode("Neg", {"a"}, {"middle"}));
  function.add_node(MakeNode("Abs", {"middle"}, {"r"}));
  model.add_function(function);
  core::builder::GraphBuilder builder(model, SchemaLookup());
  auto &local = builder.LocalFunction("F");
  ASSERT_TRUE(local.HasShape("middle"));
  EXPECT_EQ(local.GetShape("middle").Shape(), MakeShape({2, 3}));
  ASSERT_TRUE(local.HasShape("r"));
  local.MakeNode("Identity", {"r"}, {"extra"});
  local.MakeOutput("extra");
  core::builder::GraphBuilder moved(std::move(builder));
  const ModelProto exported = moved.ToModel();
  ASSERT_EQ(exported.functions().size(), 1u);
  const FunctionProto &actual = exported.functions(0);
  EXPECT_EQ(actual.name(), function.name());
  EXPECT_EQ(actual.domain(), function.domain());
  EXPECT_EQ(actual.doc_string(), function.doc_string());
  EXPECT_EQ(actual.overload(), function.overload());
  EXPECT_TRUE(std::any_of(
      actual.opset_import().begin(), actual.opset_import().end(),
      [&](const auto &opset) { return opset.version() == function.opset_import(0).version(); }));
  ASSERT_EQ(actual.attribute().size(), 1u);
  EXPECT_EQ(actual.attribute(0), "required");
  ASSERT_EQ(actual.attribute_proto().size(), 1u);
  EXPECT_EQ(actual.attribute_proto(0).SerializeAsString(), default_attribute.SerializeAsString());
  bool found_metadata = false;
  for (const auto &metadata : actual.metadata_props()) {
    if (metadata.key() == "purpose") {
      EXPECT_EQ(metadata.value(), "roundtrip");
      found_metadata = true;
    }
  }
  EXPECT_TRUE(found_metadata);
  for (const auto &expected : function.value_info()) {
    std::size_t matches = 0;
    for (const auto &value_info : actual.value_info()) {
      if (value_info.name() == expected.name()) {
        ++matches;
        EXPECT_EQ(value_info.doc_string(), expected.doc_string());
        bool found_value_metadata = false;
        for (const auto &metadata : value_info.metadata_props()) {
          if (metadata.key() == "purpose") {
            EXPECT_EQ(metadata.value(), expected.name());
            found_value_metadata = true;
          }
        }
        EXPECT_TRUE(found_value_metadata);
      }
    }
    EXPECT_EQ(matches, 1u) << expected.name();
  }
  ASSERT_EQ(actual.node().size(), 3u);
  ASSERT_EQ(actual.output().size(), 2u);
  EXPECT_EQ(actual.output(1), "extra");
}

TEST(GraphBuilder, NativeFunctionImportsIfWithUnboundAttributesOrUnknownCaptures) {
  for (const bool referenced_attribute : {false, true}) {
    SCOPED_TRACE(referenced_attribute);
    core::builder::GraphBuilder seed("g", SchemaLookup());
    seed.SetOpsetVersion("", 18);
    seed.SetOpsetVersion("local", 1);
    seed.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    seed.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
    ModelProto model = seed.ToModel();
    FunctionProto function;
    function.set_name("Select");
    function.set_domain("local");
    function.add_input("a");
    function.add_input("condition");
    function.add_output("r");
    function.add_opset("", 18);
    function.add_value_info(model.graph().input(1));
    if (referenced_attribute) {
      ValueInfoProto input = model.graph().input(0);
      input.set_name("a");
      function.add_value_info(input);
      AttributeProto default_type;
      default_type.set_name("dtype");
      default_type.set_type(AttributeProto::AttributeType::INT);
      default_type.set_i(TensorProto::DataType::DOUBLE);
      function.add_attribute_proto(default_type);
    }
    NodeProto select = MakeNode("If", {"condition"}, {"r"});
    for (const std::string name : {"then_branch", "else_branch"}) {
      AttributeProto branch;
      branch.set_name(name);
      branch.set_type(AttributeProto::AttributeType::GRAPH);
      GraphProto *graph = branch.mutable_g();
      graph->set_name(name);
      NodeProto node =
          MakeNode(referenced_attribute ? "Cast" : "Identity", {"a"}, {"branch_result"});
      if (referenced_attribute) {
        AttributeProto reference;
        reference.set_name("to");
        reference.set_type(AttributeProto::AttributeType::INT);
        reference.set_ref_attr_name("dtype");
        node.add_attribute(reference);
      }
      graph->add_node(node);
      graph->add_output()->set_name("branch_result");
      select.add_attribute(branch);
    }
    function.add_node(select);
    model.add_function(function);
    core::builder::GraphBuilder builder(model, SchemaLookup());
    EXPECT_FALSE(builder.LocalFunction("Select").HasShape("r"));
    builder.MakeNode("Select", {"X", "condition"}, {"Y"}, "local");
    ASSERT_TRUE(builder.HasShape("Y"));
    EXPECT_EQ(builder.GetShape("Y").Shape(), MakeShape({2, 3}));
    EXPECT_EQ(builder.GetShape("Y").Dtype(), referenced_attribute
                                                 ? core::symbolic::TensorType::kDouble
                                                 : core::symbolic::TensorType::kFloat);
  }
}

TEST(GraphBuilder, NativeFunctionBindsRequiredAndDefaultAttributes) {
  core::builder::GraphBuilder seed("g", SchemaLookup());
  seed.SetOpsetVersion("", 18);
  seed.SetOpsetVersion("local", 1);
  seed.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  ModelProto model = seed.ToModel();
  FunctionProto function;
  function.set_name("Convert");
  function.set_domain("local");
  function.add_input("a");
  ValueInfoProto input = model.graph().input(0);
  input.set_name("a");
  function.add_value_info(input);
  function.add_output("r");
  function.add_opset("", 18);
  AttributeProto default_type;
  default_type.set_name("dtype");
  default_type.set_type(AttributeProto::AttributeType::INT);
  default_type.set_i(TensorProto::DataType::DOUBLE);
  function.add_attribute_proto(default_type);
  NodeProto cast = MakeNode("Cast", {"a"}, {"r"});
  AttributeProto reference;
  reference.set_name("to");
  reference.set_type(AttributeProto::AttributeType::INT);
  reference.set_ref_attr_name("dtype");
  cast.add_attribute(reference);
  function.add_node(cast);
  model.add_function(function);
  FunctionProto required = function;
  required.set_name("Required");
  required.ref_attribute_proto().clear();
  required.add_attribute("dtype");
  model.add_function(required);
  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_TRUE(builder.LocalFunction("Convert").HasShape("a"));
  EXPECT_FALSE(builder.LocalFunction("Convert").HasShape("r"));
  builder.MakeNode("Convert", {"X"}, {"default"}, "local");
  EXPECT_EQ(builder.GetShape("default").Dtype(), core::symbolic::TensorType::kDouble);
  utils::RepeatedProtoField<AttributeProto> attributes;
  default_type.set_i(TensorProto::DataType::INT64);
  attributes.push_back(default_type);
  builder.MakeNode("Convert", {"X"}, {"explicit"}, "local", "", attributes);
  EXPECT_EQ(builder.GetShape("explicit").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_THROW(builder.MakeNode("Required", {"X"}, {"missing"}, "local"), std::invalid_argument);
  EXPECT_EQ(builder.Compute().Size(), builder.Nodes().size());
  builder.MakeNode("Required", {"X"}, {"provided"}, "local", "", attributes);
  EXPECT_EQ(builder.GetShape("provided").Dtype(), core::symbolic::TensorType::kInt64);
  const ModelProto exported = builder.ToModel();
  ASSERT_EQ(exported.functions(0).attribute_proto().size(), 1u);
  EXPECT_EQ(exported.functions(0).node(0).attribute(0).ref_attr_name(), "dtype");
}

TEST(GraphBuilder, NativeFunctionSeedsLateIntermediateTags) {
  core::builder::GraphBuilder seed("g", SchemaLookup());
  seed.SetOpsetVersion("", 18);
  ModelProto model = seed.ToModel();
  FunctionProto function;
  function.set_name("Tagged");
  function.set_domain("local");
  function.add_input("a");
  function.add_output("r");
  function.add_opset("", 18);
  function.add_node(MakeNode("Abs", {"a"}, {"intermediate"}));
  function.add_node(MakeNode("Abs", {"intermediate"}, {"r"}));
  auto *declared = function.add_value_info();
  declared->set_name("intermediate");
  auto *tag = declared->add_metadata_props();
  tag->set_key(core::compute::kValueTagMetadataKey);
  tag->set_value("shape");
  model.add_function(function);

  core::builder::GraphBuilder builder(model, SchemaLookup());
  auto &local = builder.LocalFunction("Tagged");
  EXPECT_EQ(local.Compute().ValueTags().at("intermediate"), "shape");
  const FunctionProto exported = local.ToFunction("local");
  const auto release =
      std::find_if(exported.node()[1].metadata_props().begin(),
                   exported.node()[1].metadata_props().end(), [](const auto &entry) {
                     return entry.key() == core::compute::kReleaseAfterShapeTagMetadataKey;
                   });
  ASSERT_NE(release, exported.node()[1].metadata_props().end());
  EXPECT_EQ(release->value(), "intermediate");
}

TEST(GraphBuilder, NativeFunctionRejectsRecursionAndWrongArity) {
  core::builder::GraphBuilder seed("g", SchemaLookup());
  seed.SetOpsetVersion("", 18);
  seed.SetOpsetVersion("local", 1);
  seed.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  ModelProto model = seed.ToModel();
  FunctionProto function;
  function.set_name("Recursive");
  function.set_domain("local");
  function.add_opset("local", 1);
  function.add_input("a");
  function.add_output("r");
  function.add_node(MakeNode("Recursive", {"a"}, {"r"}, "local"));
  model.add_function(function);
  EXPECT_THROW(core::builder::GraphBuilder imported(model, SchemaLookup()),
               core::builder::BuilderError);
  function.add_opset("", 18);
  function.ref_node().clear();
  function.add_node(MakeNode("Identity", {"a"}, {"r"}));
  model.ref_functions().clear();
  model.add_function(function);
  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_THROW(builder.MakeNode("Recursive", {"X", "X"}, {"arity"}, "local"),
               std::invalid_argument);
}

TEST(GraphBuilder, NativeUnknownRankIsNotScalar) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  ValueInfoProto unknown;
  unknown.set_name("unknown");
  unknown.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::FLOAT);
  builder.MakeInput(unknown);
  EXPECT_FALSE(builder.HasShape("unknown"));
  builder.MakeInput("scalar", core::symbolic::TensorType::kFloat, MakeShape({}));
  ASSERT_TRUE(builder.HasShape("scalar"));
  EXPECT_EQ(builder.GetShape("scalar").Shape().Rank(), 0u);
  builder.MakeNode("MatMul", {"unknown", "unknown"}, {"deferred"});
  EXPECT_FALSE(builder.HasShape("deferred"));
  EXPECT_THROW(builder.MakeNode("MatMul", {"scalar", "scalar"}, {"invalid"}),
               std::invalid_argument);
}

TEST(GraphBuilder, NativeIfInfersCapturedOutputsBeforeCastLike) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeNode("Identity", {"X"}, {"branch_result"});
    branch.MakeOutput("branch_result");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  ASSERT_TRUE(builder.HasShape("selected"));
  builder.MakeNode("CastLike", {"selected", "X"}, {"Y"});
  EXPECT_EQ(builder.GetShape("Y").Shape(), MakeShape({2, 3}));
  builder.MakeOutput("Y");
  core::builder::GraphBuilder imported(builder.ToModel(), SchemaLookup());
  EXPECT_EQ(imported.GetShape("selected").Shape(), MakeShape({2, 3}));
}

TEST(GraphBuilder, NativeSubgraphRejectsAncestorShadowing) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  auto &branch = builder.MakeSubgraph("branch");
  EXPECT_THROW(branch.MakeNode("Identity", {"X"}, {"X"}), core::builder::BuilderError);
  EXPECT_THROW(branch.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3})),
               core::builder::BuilderError);
  branch.MakeNode("Identity", {"X"}, {"Y"});
  EXPECT_EQ(branch.GetShape("Y").Shape(), MakeShape({2, 3}));
}

TEST(GraphBuilder, NativeCleanupPrunesInitializersWithoutNodes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("public", core::symbolic::TensorType::kFloat, MakeShape({1}));
  builder.MakeInitializer(MakeInitializer<float>("public", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("output", {1}, {2.0f}));
  builder.MakeInitializer(MakeInitializer<float>("orphan", {1}, {3.0f}));
  builder.MakeOutput("output");
  EXPECT_EQ(builder.RemoveUnusedNodes(), 0u);
  ASSERT_EQ(builder.Initializers().size(), 2u);
  EXPECT_EQ(builder.Initializers()[0].name(), "public");
  EXPECT_EQ(builder.Initializers()[1].name(), "output");
  const GraphProto graph = builder.ToGraph();
  EXPECT_EQ(graph.value_info().size(), 0u);
  EXPECT_EQ(graph.input().size(), 1u);
}

TEST(GraphBuilder, NativeCleanupRetainsInitializerCapturedByLiveBranch) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeInitializer(MakeInitializer<float>("captured", {1}, {1.0f}));
  builder.MakeInitializer(MakeInitializer<float>("orphan", {1}, {2.0f}));
  builder.MakeNode("Neg", {"orphan"}, {"dead"});
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeNode("Identity", {"captured"}, {"branch_result"});
    branch.MakeOutput("branch_result");
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  builder.MakeNode("If", {"condition"}, {"Y"}, "", "", attributes);
  builder.MakeOutput("Y");
  EXPECT_EQ(builder.RemoveUnusedNodes(), 1u);
  ASSERT_EQ(builder.Initializers().size(), 1u);
  EXPECT_EQ(builder.Initializers()[0].name(), "captured");
  EXPECT_EQ(builder.ToGraph().value_info().size(), 0u);
}

TEST(GraphBuilder, NativeFunctionSnapshotsRefreshAfterMutations) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("local", 1);
  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeNode("Neg", {"a"}, {"dead"});
  function.MakeOutput("r");
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("F", {"X"}, {"result"}, "local");
  const FunctionProto definition = *builder.Shapes().GetLocalFunction("local:F");
  ASSERT_EQ(definition.node().size(), 2u);

  auto &growing_function = builder.MakeLocalFunction("Identity", "local");
  growing_function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  std::string previous = "a";
  for (int i = 0; i < 16; ++i) {
    const std::string output = "inner_" + std::to_string(i);
    growing_function.MakeNode("Identity", {previous}, {output});
    previous = output;
    builder.MakeNode("Identity", {"X"}, {"ordinary_" + std::to_string(i)});
    EXPECT_EQ(builder.Shapes().GetLocalFunction("local:F")->node().size(), 2u);
    EXPECT_TRUE(builder.Shapes().HasLocalFunction("local:Identity"));
  }
  growing_function.MakeOutput(previous);
  EXPECT_TRUE(builder.Shapes().HasLocalFunction("local:Identity"));
  builder.MakeNode("Identity", {"X"}, {"from_growing"}, "local");
  EXPECT_EQ(builder.GetShape("from_growing").Shape(), MakeShape({2, 3}));
  ASSERT_NE(builder.Shapes().GetLocalFunction("local:Identity"), nullptr);
  EXPECT_EQ(builder.Shapes().GetLocalFunction("local:Identity")->node().size(), 16u);

  EXPECT_EQ(function.RemoveUnusedNodes(), 1u);
  EXPECT_EQ(builder.Shapes().GetLocalFunction("local:F")->node().size(), 1u);
  builder.MakeNode("Add", {"result", "X"}, {"ordinary_after_edit"});
  EXPECT_EQ(builder.Shapes().GetLocalFunction("local:F")->node().size(), 1u);
  builder.MakeNode("F", {"X"}, {"after_edit"}, "local");
  ASSERT_NE(builder.Shapes().GetLocalFunction("local:F"), nullptr);
  EXPECT_EQ(builder.Shapes().GetLocalFunction("local:F")->node().size(), 1u);
  EXPECT_EQ(builder.GetShape("after_edit").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(definition.node().size(), 2u);
}

TEST(GraphBuilder, NativeDemandRefreshDiscardsRemovedFunctions) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("local", 1);
  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeNode("F", {"X"}, {"result"}, "local");
  builder.MakeOutput("result");
  const FunctionProto snapshot = *builder.Shapes().GetLocalFunction("local:F");
  EXPECT_EQ(builder.InlineLocalFunctions(), 1u);
  EXPECT_FALSE(builder.HasLocalFunction("F"));
  builder.MakeNode("F", {"X"}, {"unregistered"}, "local");
  EXPECT_FALSE(builder.HasShape("unregistered"));
  EXPECT_FALSE(builder.Shapes().HasLocalFunction("local:F"));
  EXPECT_EQ(snapshot.name(), "F");
}

TEST(GraphBuilder, NativeGraphReferencesRefreshLateFunctionDefinitions) {
  core::builder::GraphBuilder builder("g", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("local", 1);
  builder.MakeInput("X", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  utils::RepeatedProtoField<AttributeProto> attributes;
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto &branch = builder.MakeSubgraph(name);
    branch.MakeNode("F", {"X"}, {"branch_result"}, "local");
    branch.MakeOutput("branch_result");
    EXPECT_FALSE(branch.HasShape("branch_result"));
    AttributeProto reference;
    reference.set_name(name + "_ref");
    reference.set_type(AttributeProto::AttributeType::STRING);
    reference.set_s(name);
    attributes.push_back(reference);
  }
  auto &function = builder.MakeLocalFunction("F", "local");
  function.MakeInput("a", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
  function.MakeNode("Identity", {"a"}, {"r"});
  function.MakeOutput("r");
  EXPECT_TRUE(builder.Shapes().HasLocalFunction("local:F"));
  EXPECT_EQ(builder.Subgraph("then_branch").GetShape("branch_result").Shape(), MakeShape({2, 3}));
  builder.MakeNode("If", {"condition"}, {"selected"}, "", "", attributes);
  EXPECT_EQ(builder.GetShape("selected").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(builder.Subgraph("then_branch").GetShape("branch_result").Shape(), MakeShape({2, 3}));
  const std::string snapshot = builder.Shapes().GetLocalFunction("local:F")->SerializeAsString();
  builder.MakeNode("CastLike", {"selected", "X"}, {"Y"});
  EXPECT_EQ(builder.GetShape("Y").Shape(), MakeShape({2, 3}));
  EXPECT_EQ(builder.Shapes().GetLocalFunction("local:F")->SerializeAsString(), snapshot);
}

TEST(GraphBuilder, NativeDefaultInitializerValidatesDeclaredTensor) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  const std::vector<TensorProto> incompatible = {
      MakeInitializer<int64_t>("x", {2}, {1, 2}), MakeInitializer<float>("x", {1, 2}, {1.0f, 2.0f}),
      MakeInitializer<float>("x", {3}, {1.0f, 2.0f, 3.0f})};
  for (const TensorProto &tensor : incompatible) {
    EXPECT_THROW(builder.MakeInitializer(tensor), core::builder::BuilderError);
    EXPECT_TRUE(builder.Initializers().empty());
    EXPECT_EQ(builder.GetShape("x").Dtype(), core::symbolic::TensorType::kFloat);
    EXPECT_EQ(builder.GetShape("x").Shape(), MakeShape({2}));
  }
  const TensorProto compatible = MakeInitializer<float>("x", {2}, {1.0f, 2.0f});
  EXPECT_EQ(builder.MakeInitializer(compatible), "x");
  ASSERT_EQ(builder.Initializers().size(), 1u);
  EXPECT_THROW(builder.MakeInitializer(compatible), core::builder::BuilderError);
}

TEST(GraphBuilder, NativeDefaultInitializerAllowsSymbolicAndUnknownDimensions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  ValueInfoProto input;
  input.set_name("x");
  auto *type = input.mutable_type()->mutable_tensor_type();
  type->set_elem_type(TensorProto::DataType::FLOAT);
  type->mutable_shape()->add_dim()->set_dim_param("batch");
  type->mutable_shape()->add_dim()->set_dim_param("");
  type->mutable_shape()->add_dim()->set_dim_value(3);
  builder.MakeInput(input);
  const auto expected_shape = builder.GetShape("x").Shape();
  EXPECT_EQ(
      builder.MakeInitializer(MakeInitializer<float>("x", {2, 4, 3}, std::vector<float>(24, 1.0f))),
      "x");
  EXPECT_EQ(builder.GetShape("x").Shape(), expected_shape);
  const auto &declared = builder.Inputs()[0].type().tensor_type().shape();
  EXPECT_EQ(declared.dim(0).dim_param(), "batch");
  EXPECT_FALSE(declared.dim(1).has_dim_value());
  EXPECT_TRUE(declared.dim(1).has_dim_param());
  EXPECT_TRUE(declared.dim(1).dim_param().empty());
  EXPECT_EQ(declared.dim(2).dim_value(), 3);
}

TEST(GraphBuilder, NativeDefaultInitializerAllowsUnknownRankAndType) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  ValueInfoProto input;
  input.set_name("unknown_rank");
  input.mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::DataType::FLOAT);
  builder.MakeInput(input);
  EXPECT_EQ(builder.MakeInitializer(MakeInitializer<float>("unknown_rank", {2}, {1.0f, 2.0f})),
            "unknown_rank");
  EXPECT_FALSE(builder.Inputs()[0].type().tensor_type().has_shape());
  EXPECT_FALSE(builder.Compute().Shapes().Has("unknown_rank"));
  ValueInfoProto untyped;
  untyped.set_name("untyped");
  builder.MakeInput(untyped);
  EXPECT_EQ(builder.MakeInitializer(MakeInitializer<int64_t>("untyped", {1}, {1})), "untyped");
  builder.MakeInput("scalar", core::symbolic::TensorType::kFloat, MakeShape({}));
  EXPECT_THROW(builder.MakeInitializer(MakeInitializer<float>("scalar", {1}, {1.0f})),
               core::builder::BuilderError);
  EXPECT_EQ(builder.MakeInitializer(MakeInitializer<float>("scalar", {}, {1.0f})), "scalar");
}

TEST(GraphBuilder, NativeDefaultInitializerRejectsNonTensorInput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  ValueInfoProto input;
  input.set_name("sequence");
  input.mutable_type()
      ->mutable_sequence_type()
      ->mutable_elem_type()
      ->mutable_tensor_type()
      ->set_elem_type(TensorProto::DataType::FLOAT);
  builder.MakeInput(input);
  EXPECT_THROW(builder.MakeInitializer(MakeInitializer<float>("sequence", {1}, {1.0f})),
               core::builder::BuilderError);
  EXPECT_TRUE(builder.Initializers().empty());
}

namespace {

StructTypeProto BuilderRecord(uint64_t id = 1) {
  StructTypeProto type;
  type.set_type_id(id);
  type.set_name("record");
  auto *field = type.mutable_structure()->add_field();
  field->set_name("scale");
  auto *tensor = field->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape();
  field = type.mutable_structure()->add_field();
  field->set_name("codes");
  auto *packing = field->mutable_type()->mutable_struct_type()->mutable_bit_packing();
  packing->set_dimension(uint64_t{2});
  auto *component = packing->add_component();
  component->set_name("nibble");
  component->set_bit_width(uint32_t{4});
  field = type.mutable_structure()->add_field();
  field->set_name("offsets");
  auto *array = field->mutable_type()->mutable_struct_type()->mutable_array();
  array->set_dimension(uint64_t{3});
  tensor = array->mutable_element_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::UINT8);
  tensor->mutable_shape();
  field = type.mutable_structure()->add_field();
  field->set_name("constant");
  *field->mutable_constant() = MakeInitializer<float>("", {}, {2.f});
  return type;
}

EncodedValueProto BuilderEncoded(const std::string &name, uint64_t id = 1, std::size_t bytes = 16) {
  EncodedValueProto value;
  value.set_name(name);
  value.mutable_struct_type()->set_type_ref(id);
  value.set_raw_data(std::string(bytes, '\x2a'));
  return value;
}

ValueInfoProto BuilderStructuredInfo(const std::string &name, uint64_t id = 1) {
  ValueInfoProto info;
  info.set_name(name);
  info.mutable_type()->mutable_struct_type()->set_type_ref(id);
  return info;
}

} // namespace

TEST(GraphBuilderStructured, NativeRoundtripPreservesDeclarationsAndMetadata) {
  ModelProto model;
  model.set_ir_version(10);
  model.set_producer_name("original");
  model.set_doc_string("model documentation");
  model.add_opset("", 23);
  model.ref_struct_types().push_back(BuilderRecord());
  auto *graph = model.mutable_graph();
  graph->set_name("native");
  graph->set_doc_string("graph documentation");
  *graph->add_extended_initializer()->mutable_encoded_value() = BuilderEncoded("weight");
  graph->add_value_info(BuilderStructuredInfo("weight"))->set_doc_string("encoded descriptor");
  graph->add_value_info(BuilderStructuredInfo("result"));
  graph->add_initializer(MakeInitializer<float>("scale", {}, {2.f}));
  auto *annotation = graph->add_quantization_annotation();
  annotation->set_tensor_name("weight");
  auto *parameter = annotation->add_quant_parameter_tensor_names();
  parameter->set_key("SCALE_TENSOR");
  parameter->set_value("scale");
  auto *node = graph->add_node();
  node->set_op_type("Identity");
  node->add_input("weight");
  node->add_output("result");
  node->set_doc_string("node documentation");
  auto *metadata = node->add_metadata_props();
  metadata->set_key("application");
  metadata->set_value("retained");
  graph->add_output(BuilderStructuredInfo("result"));
  graph->ref_output()[0].mutable_type()->set_denotation("DECLARED_RECORD");
  core::builder::GraphBuilder builder(model, SchemaLookup());
  EXPECT_EQ(builder.Shapes().GetEncodedLayout("weight").element_bits, 64u);
  EXPECT_EQ(builder.Shapes().GetEncodedLayout("weight").record_count, 2u);
  auto exported = builder.ToModel();
  EXPECT_EQ(exported.ir_version(), 10);
  EXPECT_EQ(exported.producer_name(), "original");
  EXPECT_EQ(exported.doc_string(), model.doc_string());
  EXPECT_EQ(exported.graph().doc_string(), graph->doc_string());
  ASSERT_EQ(exported.graph().quantization_annotation().size(), 1u);
  EXPECT_EQ(exported.graph().quantization_annotation()[0].SerializeAsString(),
            annotation->SerializeAsString());
  EXPECT_TRUE(std::any_of(exported.graph().value_info().begin(),
                          exported.graph().value_info().end(), [](const auto &value) {
                            return value.name() == "weight" &&
                                   value.doc_string() == "encoded descriptor";
                          }));
  EXPECT_EQ(exported.graph().node()[0].doc_string(), node->doc_string());
  EXPECT_TRUE(std::any_of(exported.graph().node()[0].metadata_props().begin(),
                          exported.graph().node()[0].metadata_props().end(), [](const auto &entry) {
                            return entry.key() == "application" && entry.value() == "retained";
                          }));
  ASSERT_EQ(exported.struct_types().size(), 1u);
  EXPECT_EQ(exported.struct_types()[0].SerializeAsString(),
            model.struct_types()[0].SerializeAsString());
  ASSERT_EQ(exported.graph().extended_initializer().size(), 1u);
  EXPECT_EQ(exported.graph().extended_initializer()[0].encoded_value().SerializeAsString(),
            graph->extended_initializer()[0].encoded_value().SerializeAsString());
  EXPECT_EQ(exported.graph().output()[0].type().SerializeAsString(),
            graph->output()[0].type().SerializeAsString());
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(exported);
  context.ApplyInferredShapesToModel(exported);
  EXPECT_EQ(exported.graph().output()[0].type().denotation(), "DECLARED_RECORD");
  EXPECT_NO_THROW(VerifyModel(exported));
  EXPECT_THROW(SerializeModelToOrtFlatbuffers(exported, {}), std::invalid_argument);
  EXPECT_THROW(builder.ToStandardModel(), core::builder::BuilderError);
  EXPECT_THROW(builder.ToFunction(), core::builder::BuilderError);
}

TEST(GraphBuilderStructured, NativeFunctionReplaysLateStructuredDeclarations) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset("", 23);
  model.add_opset("custom", 1);
  model.ref_struct_types().push_back(BuilderRecord());
  model.mutable_graph()->set_name("g");
  FunctionProto function;
  function.set_name("Structured");
  function.set_domain("local");
  function.add_output("result");
  function.add_opset("", 23);
  function.add_opset("custom", 1);
  function.add_node(MakeNode("Unknown", {}, {"intermediate"}, "custom"));
  function.add_node(MakeNode("Identity", {"intermediate"}, {"result"}));
  function.add_value_info(BuilderStructuredInfo("intermediate"));
  model.add_function(function);

  core::builder::GraphBuilder builder(model, SchemaLookup());
  const auto &local = builder.LocalFunction("Structured");
  ASSERT_TRUE(local.Shapes().HasType("intermediate"));
  ASSERT_TRUE(local.Shapes().HasType("result"));
  EXPECT_TRUE(local.Shapes().GetType("result").has_struct_type());
  EXPECT_EQ(local.Shapes().GetType("result").struct_type().type_ref(), 1u);
}

TEST(GraphBuilderStructured, RetainsStructuredInputAndIdentityOutput) {
  core::builder::GraphBuilder builder("typed", SchemaLookup(), {{"", 28}});
  builder.MakeStructType(BuilderRecord());
  builder.MakeInput(BuilderStructuredInfo("x"));
  builder.MakeNode("Identity", {"x"}, {"y"});
  builder.MakeOutput("y");
  EXPECT_TRUE(builder.Shapes().HasType("y"));
  const auto model = builder.ToModel();
  EXPECT_TRUE(model.graph().input()[0].type().has_struct_type());
  ASSERT_TRUE(model.graph().output()[0].type().has_struct_type());
  EXPECT_EQ(model.graph().output()[0].type().struct_type().type_ref(), 1u);
}

TEST(GraphBuilderStructured, InfersConstructionMetadataExactlyOnce) {
  core::builder::GraphBuilder builder("typed", SchemaLookup(), {{"", 28}});
  builder.MakeStructType(BuilderRecord());
  builder.MakeInput(BuilderStructuredInfo("x"));
  auto &compute = const_cast<core::compute::ComputeContext &>(builder.Compute());
  compute.Shapes().set_events_enabled(true);
  compute.set_events_enabled(true);

  builder.MakeNode("Identity", {"x"}, {"y"});
  builder.MakeOutput("y");
  const auto count_shape_inference = [&]() {
    return std::count_if(compute.Shapes().Events().begin(), compute.Shapes().Events().end(),
                         [](const auto &event) {
                           return event.action == core::shapes::ShapeEventAction::kComputeNode;
                         });
  };
  EXPECT_EQ(count_shape_inference(), 1);
  EXPECT_TRUE(compute.Events().empty());

  const auto first = builder.ToModel();
  EXPECT_EQ(count_shape_inference(), 1);
  EXPECT_TRUE(compute.Events().empty());
  EXPECT_TRUE(first.graph().output()[0].type().has_struct_type());

  const auto second = builder.ToModel();
  EXPECT_EQ(count_shape_inference(), 1);
  EXPECT_TRUE(compute.Events().empty());
  EXPECT_EQ(second.SerializeAsString(), first.SerializeAsString());
}

TEST(GraphBuilderStructured, RejectsBadPayloadAndNamesWithoutMutation) {
  core::builder::GraphBuilder builder("invalid", SchemaLookup(), {{"", 28}});
  builder.MakeStructType(BuilderRecord());
  EXPECT_THROW(builder.MakeEncodedInitializer(BuilderEncoded("bad", 1, 7)), std::invalid_argument);
  EXPECT_FALSE(builder.HasName("bad"));
  EXPECT_TRUE(builder.ExtendedInitializers().empty());
  EXPECT_THROW(builder.MakeEncodedInitializer(BuilderEncoded("missing", 999)),
               std::invalid_argument);
  EXPECT_FALSE(builder.HasName("missing"));
  builder.MakeEncodedInitializer(BuilderEncoded("weight"));
  EXPECT_THROW(builder.MakeEncodedInitializer(BuilderEncoded("weight")),
               core::builder::BuilderError);
  EXPECT_THROW(builder.MakeInitializer(MakeInitializer<float>("weight", {}, {1.f})),
               core::builder::BuilderError);
  EXPECT_THROW(builder.MakeNode("Identity", {"weight"}, {"weight"}), core::builder::BuilderError);
  EXPECT_THROW(builder.MakeStructType(BuilderRecord()), std::invalid_argument);
  EXPECT_EQ(builder.Shapes().StructTypes().size(), 1u);
  builder.MakeStructType(BuilderRecord(2));
  EXPECT_THROW(builder.MakeOutput(BuilderStructuredInfo("weight", 2)), core::builder::BuilderError);
  EXPECT_THROW(builder.MakeOutput(BuilderStructuredInfo("weight", 999)), std::invalid_argument);
  EXPECT_THROW(builder.MakeOutput("weight", core::symbolic::TensorType::kFloat, MakeShape({4})),
               core::builder::BuilderError);
  EXPECT_TRUE(builder.Outputs().empty());
  EXPECT_EQ(builder.Shapes().GetEncodedLayout("weight").record_count, 2u);
}

TEST(GraphBuilderStructured, DeduplicatesOnlyCompleteSemanticIdentity) {
  core::builder::GraphBuilder builder("dedup", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("custom", 1);
  builder.MakeStructType(BuilderRecord(1));
  auto distinct = BuilderRecord(2);
  *distinct.mutable_structure()->ref_field()[3].mutable_constant() =
      MakeInitializer<float>("", {}, {3.f});
  builder.MakeStructType(distinct);
  builder.MakeEncodedInitializer(BuilderEncoded("first"));
  builder.MakeEncodedInitializer(BuilderEncoded("duplicate"));
  builder.MakeEncodedInitializer(BuilderEncoded("different_type", 2));
  builder.MakeEncodedInitializer(BuilderEncoded("short", 1, 8));
  builder.MakeEncodedInitializer(BuilderEncoded("public_output"));
  builder.MakeNode("Consume", {"duplicate", "different_type"}, {"result"}, "custom");
  builder.MakeOutput("result");
  builder.MakeOutput(BuilderStructuredInfo("public_output"));
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 1u);
  ASSERT_EQ(builder.ExtendedInitializers().size(), 4u);
  EXPECT_EQ(builder.Shapes().GetEncodedLayout("first").record_count, 2u);
  EXPECT_EQ(builder.Shapes().GetEncodedLayout("short").record_count, 1u);
  EXPECT_EQ(builder.Nodes()[0].input(0), "first");
  EXPECT_EQ(builder.Nodes()[0].input(1), "different_type");
  EXPECT_FALSE(builder.Shapes().HasEncodedValue("duplicate"));
  EXPECT_EQ(builder.ToModel().graph().extended_initializer().size(), 4u);
}

TEST(GraphBuilderStructured, PrunesEncodedInitializersAndKeepsCapturedValues) {
  core::builder::GraphBuilder builder("captures", SchemaLookup(), {{"", 28}});
  builder.SetOpsetVersion("custom", 1);
  builder.MakeStructType(BuilderRecord());
  builder.MakeEncodedInitializer(BuilderEncoded("live"));
  builder.MakeEncodedInitializer(BuilderEncoded("dead"));
  auto &child = builder.MakeSubgraph("body");
  child.MakeNode("Identity", {"live"}, {"local_live"});
  child.MakeOutput(BuilderStructuredInfo("local_live"));
  AttributeProto attr;
  attr.set_name("body_ref");
  attr.set_type(AttributeProto::STRING);
  attr.set_s("body");
  utils::RepeatedProtoField<AttributeProto> attrs;
  attrs.push_back(attr);
  builder.MakeNode("Capture", {}, {"result"}, "custom", "", attrs);
  builder.MakeOutput("result");
  EXPECT_EQ(builder.RemoveUnusedNodes(), 0u);
  ASSERT_EQ(builder.ExtendedInitializers().size(), 1u);
  EXPECT_EQ(builder.ExtendedInitializers()[0].name(), "live");
  EXPECT_FALSE(builder.Shapes().HasEncodedValue("dead"));
  const auto model = builder.ToModel();
  EXPECT_EQ(model.graph().node()[0].attribute()[0].g().output()[0].name(), "local_live");
}

TEST(GraphBuilderStructured, NestedDedupRewritesCapturesAndOwnsCatalogue) {
  core::builder::GraphBuilder builder("nested", SchemaLookup(), {{"", 28}});
  builder.MakeStructType(BuilderRecord());
  builder.MakeEncodedInitializer(BuilderEncoded("outer"));
  auto &child = builder.MakeSubgraph("body");
  child.MakeEncodedInitializer(BuilderEncoded("inner"));
  child.MakeNode("Identity", {"inner"}, {"result"});
  child.MakeOutput("result");
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 1u);
  EXPECT_TRUE(child.ExtendedInitializers().empty());
  EXPECT_EQ(child.Nodes()[0].input(0), "outer");
  EXPECT_FALSE(child.Shapes().HasEncodedValue("inner"));
  EXPECT_EQ(child.Shapes().GetEncodedLayout("outer").element_bits, 64u);
  core::shapes::ShapesContext snapshot;
  snapshot.SetStructTypes(child.Shapes().StructTypes());
  snapshot.CopyValueFrom("outer", child.Shapes(), "outer");
  builder.MakeStructType(BuilderRecord(2));
  EXPECT_EQ(snapshot.StructTypes().size(), 1u);
  EXPECT_EQ(child.Shapes().StructTypes().size(), 2u);
  EXPECT_EQ(snapshot.GetEncodedLayout("outer").root->type_id(), 1u);
  EXPECT_NO_THROW(child.ToGraph());
}

TEST(GraphBuilderStructured, RetainsPayloadOwnerAcrossImportMoveAndExport) {
  bool released = false;
  ModelProto exported;
  {
    ModelProto model;
    model.add_opset("", 23);
    model.ref_struct_types().push_back(BuilderRecord());
    auto value = BuilderEncoded("weight");
    auto bytes = std::make_shared<std::vector<uint8_t>>(16, 42);
    value.set_raw_data_with_deleter(bytes->data(), bytes->size(),
                                    [bytes, &released]() { released = true; });
    model.mutable_graph()->set_name("owner");
    *model.mutable_graph()->add_extended_initializer()->mutable_encoded_value() = value;
    model.mutable_graph()->add_output(BuilderStructuredInfo("weight"));
    core::builder::GraphBuilder imported(model, SchemaLookup());
    model.Clear();
    value.Clear();
    bytes.reset();
    core::builder::GraphBuilder moved(std::move(imported));
    EXPECT_FALSE(released);
    exported = moved.ToModel();
    EXPECT_EQ(exported.graph().extended_initializer()[0].encoded_value().raw_data().size(), 16u);
    EXPECT_FALSE(released);
  }
  EXPECT_FALSE(released);
  exported.Clear();
  EXPECT_TRUE(released);
}

TEST(GraphBuilderStructured, ExternalPayloadLengthIsValidatedAndRetained) {
  core::builder::GraphBuilder builder("external", SchemaLookup());
  builder.MakeStructType(BuilderRecord());
  auto value = BuilderEncoded("external");
  value.clear_raw_data();
  value.set_data_location(TensorProto::EXTERNAL);
  auto *entry = value.add_external_data();
  entry->set_key("location");
  entry->set_value("records.bin");
  entry = value.add_external_data();
  entry->set_key("length");
  entry->set_value("15");
  EXPECT_THROW(builder.MakeEncodedInitializer(value), std::invalid_argument);
  entry->set_value("16");
  builder.MakeEncodedInitializer(value);
  builder.MakeOutput(BuilderStructuredInfo("external"));
  EXPECT_EQ(builder.Shapes().GetEncodedLayout("external").record_count, 2u);
  EXPECT_FALSE(builder.Shapes().GetEncodedLayout("external").content_verified);
  const auto graph = builder.ToGraph();
  EXPECT_EQ(graph.extended_initializer()[0].encoded_value().SerializeAsString(),
            value.SerializeAsString());
}

TEST(GraphBuilderStructured, NestedGraphExportRetainsPayloadOwner) {
  bool released = false;
  GraphProto exported;
  {
    core::builder::GraphBuilder builder("nested_owner", SchemaLookup(), {{"", 28}});
    builder.SetOpsetVersion("custom", 1);
    builder.MakeStructType(BuilderRecord());
    auto &child = builder.MakeSubgraph("body");
    auto value = BuilderEncoded("weight");
    auto bytes = std::make_shared<std::vector<uint8_t>>(16, 42);
    value.set_raw_data_with_deleter(bytes->data(), bytes->size(),
                                    [bytes, &released]() { released = true; });
    child.MakeEncodedInitializer(value);
    child.MakeOutput(BuilderStructuredInfo("weight"));
    utils::RepeatedProtoField<AttributeProto> attributes;
    auto &attribute = attributes.add();
    attribute.set_name("body_ref");
    attribute.set_type(AttributeProto::STRING);
    attribute.set_s("body");
    builder.MakeNode("Capture", {}, {"result"}, "custom", "", attributes);
    builder.MakeOutput(BuilderStructuredInfo("result"));
    exported = builder.ToGraph();
  }
  EXPECT_FALSE(released);
  ASSERT_EQ(exported.node()[0].attribute()[0].g().extended_initializer().size(), 1u);
  exported.Clear();
  EXPECT_TRUE(released);
}

TEST(GraphBuilderStructured, StandardExportRejectsInlineAndNestedTypes) {
  core::builder::GraphBuilder standard("standard", SchemaLookup());
  standard.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  standard.MakeOutput("x");
  EXPECT_NO_THROW(standard.ToStandardModel());
  core::builder::GraphBuilder structured("inline", SchemaLookup());
  auto info = BuilderStructuredInfo("x");
  auto inline_type = BuilderRecord();
  inline_type.clear_type_id();
  *info.mutable_type()->mutable_sequence_type()->mutable_elem_type()->mutable_struct_type() =
      inline_type;
  structured.MakeInput(info);
  structured.MakeOutput(info);
  EXPECT_THROW(structured.ToStandardModel(), core::builder::BuilderError);
}

TEST(GraphBuilderStructured, ReordersNodesAndRebuildsAfterIdentityRemoval) {
  core::builder::GraphBuilder builder("sort", SchemaLookup(), {{"", 28}});
  builder.MakeStructType(BuilderRecord());
  builder.MakeEncodedInitializer(BuilderEncoded("weight"));
  builder.MakeNode("Identity", {"weight"}, {"middle"});
  builder.MakeNode("Identity", {"middle"}, {"last"});
  builder.MakeOutput("last");
  auto &nodes = const_cast<utils::RepeatedProtoField<NodeProto> &>(builder.Nodes());
  std::swap(nodes[0], nodes[1]);
  auto graph = builder.ToGraph();
  ASSERT_EQ(graph.node().size(), 2u);
  EXPECT_EQ(graph.node()[0].input(0), "weight");
  ASSERT_TRUE(graph.output()[0].type().has_struct_type());
  EXPECT_EQ(builder.RemoveIdentityNodes(), 1u);
  EXPECT_FALSE(builder.Shapes().HasType("middle"));
  ASSERT_TRUE(builder.Shapes().HasEncodedValue("last"));
  EXPECT_EQ(builder.Shapes().GetEncodedLayout("last").record_count, 2u);
  graph = builder.ToGraph();
  ASSERT_EQ(graph.node().size(), 1u);
  EXPECT_EQ(graph.node()[0].input(0), "weight");
  EXPECT_EQ(builder.Compute().ValueTags().at("weight"), "weight");
  EXPECT_TRUE(builder.Compute().IsConstantValue("weight"));
  const auto not_used = std::find_if(
      graph.node()[0].metadata_props().begin(), graph.node()[0].metadata_props().end(),
      [](const auto &entry) { return entry.key() == core::compute::kNotUsedAfterMetadataKey; });
  ASSERT_NE(not_used, graph.node()[0].metadata_props().end());
  EXPECT_EQ(not_used->value(), "weight");
}

TEST(GraphBuilderStructured, DoesNotDeduplicateDifferentInlineConstantsOrLogicalTypes) {
  core::builder::GraphBuilder builder("semantics", SchemaLookup());
  auto first = BuilderEncoded("first");
  auto declaration = BuilderRecord();
  declaration.clear_type_id();
  *first.mutable_struct_type() = declaration;
  auto different_constant = first;
  different_constant.set_name("constant");
  *different_constant.mutable_struct_type()
       ->mutable_structure()
       ->ref_field()[3]
       .mutable_constant() = MakeInitializer<float>("", {}, {3.f});
  auto logical = first;
  logical.set_name("logical");
  auto *tensor = logical.mutable_logical_type()->mutable_tensor_type();
  tensor->set_elem_type(TensorProto::FLOAT);
  tensor->mutable_shape()->add_dim()->set_dim_value(int64_t{4});
  builder.MakeEncodedInitializer(first);
  builder.MakeEncodedInitializer(different_constant);
  builder.MakeEncodedInitializer(logical);
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 0u);
  EXPECT_EQ(builder.ExtendedInitializers().size(), 3u);
}

TEST(GraphBuilderStructured, DefaultsRemainOverridableAndProtectedFromDeduplication) {
  core::builder::GraphBuilder builder("defaults", SchemaLookup(), {{"", 28}});
  builder.MakeStructType(BuilderRecord());
  builder.MakeStructType(BuilderRecord(2));
  builder.MakeInput(BuilderStructuredInfo("input"));
  EXPECT_THROW(builder.MakeEncodedInitializer(BuilderEncoded("input", 2)),
               core::builder::BuilderError);
  EXPECT_TRUE(builder.ExtendedInitializers().empty());
  builder.MakeEncodedInitializer(BuilderEncoded("input"));
  builder.MakeEncodedInitializer(BuilderEncoded("constant"));
  builder.MakeNode("Identity", {"input"}, {"output"});
  builder.MakeOutput("output");
  EXPECT_FALSE(builder.Shapes().HasEncodedValue("input"));
  EXPECT_FALSE(builder.Shapes().HasEncodedValue("output"));
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 0u);
  const auto model = builder.ToModel();
  EXPECT_FALSE(builder.Shapes().HasEncodedValue("input"));
  EXPECT_EQ(model.graph().extended_initializer().size(), 2u);
  core::shapes::ShapesContext context;
  context.ComputeShapeModel(model);
  EXPECT_FALSE(context.HasEncodedValue("input"));
  EXPECT_FALSE(context.HasEncodedValue("output"));
  EXPECT_TRUE(context.HasEncodedValue("constant"));
}

TEST(GraphBuilderStructured, LocalFunctionUsesModelCatalogueButCannotExportItAlone) {
  core::builder::GraphBuilder builder("functions", SchemaLookup(), {{"", 28}});
  builder.MakeStructType(BuilderRecord());
  builder.SetOpsetVersion("custom", 1);
  auto &function = builder.MakeLocalFunction("Forward", "custom");
  function.SetOpsetVersion("", 23);
  function.MakeInput(BuilderStructuredInfo("arg"));
  function.MakeNode("Identity", {"arg"}, {"result"});
  function.MakeOutput("result");
  builder.MakeInput(BuilderStructuredInfo("input"));
  builder.MakeNode("Forward", {"input"}, {"output"}, "custom");
  builder.MakeOutput("output");
  EXPECT_THROW(function.ToFunction("custom"), core::builder::BuilderError);
  const auto model = builder.ToModel();
  ASSERT_EQ(model.functions().size(), 1u);
  EXPECT_EQ(model.struct_types().size(), 1u);
  EXPECT_TRUE(model.graph().output()[0].type().has_struct_type());
  EXPECT_NO_THROW(VerifyModel(model));
}

TEST(GraphBuilderStructured, RejectsNonByteAlignedRootAndUnresolvedInput) {
  core::builder::GraphBuilder builder("invalid_inline", SchemaLookup());
  auto value = BuilderEncoded("nibble");
  value.mutable_struct_type()->clear_type_ref();
  auto *packing = value.mutable_struct_type()->mutable_bit_packing();
  packing->set_dimension(uint64_t{1});
  auto *component = packing->add_component();
  component->set_name("bits");
  component->set_bit_width(uint32_t{4});
  EXPECT_THROW(builder.MakeEncodedInitializer(value), std::invalid_argument);
  EXPECT_FALSE(builder.HasName("nibble"));
  EXPECT_THROW(builder.MakeInput(BuilderStructuredInfo("unknown", 999)), std::invalid_argument);
  EXPECT_FALSE(builder.HasName("unknown"));
}

TEST(GraphBuilderStructured, DoesNotConstantFoldEncodedBytesAsLogicalTensor) {
  core::builder::GraphBuilder builder("no_dense_fold", SchemaLookup(), {{"", 28}});
  builder.MakeStructType(BuilderRecord());
  auto value = BuilderEncoded("weight");
  auto *logical = value.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(int64_t{4});
  builder.MakeEncodedInitializer(value);
  builder.MakeNode("Identity", {"weight"}, {"output"});
  builder.MakeOutput("output");
  EXPECT_EQ(builder.ConstantFold(), 0u);
  EXPECT_EQ(builder.Nodes().size(), 1u);
  EXPECT_EQ(builder.ExtendedInitializers().size(), 1u);
  EXPECT_TRUE(builder.Initializers().empty());
  EXPECT_TRUE(builder.Shapes().HasEncodedValue("output"));
}

TEST(GraphBuilderStructured, AffineDeduplicationIncludesScaleNotOnlyCodes) {
  core::builder::GraphBuilder builder("affine", SchemaLookup());
  builder.SetOpsetVersion("", 23);
  EncodedValueProto first;
  first.set_name("half_scale");
  first.mutable_affine()->set_storage_type(TensorProto::INT8);
  *first.mutable_affine()->mutable_scale() = MakeInitializer<float>("", {}, {0.5f});
  auto *logical = first.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(int64_t{4});
  first.set_raw_data(std::string(4, '\x2a'));
  auto second = first;
  second.set_name("unit_scale");
  *second.mutable_affine()->mutable_scale() = MakeInitializer<float>("", {}, {1.f});
  auto duplicate = first;
  duplicate.set_name("duplicate");
  builder.MakeEncodedInitializer(first);
  builder.MakeEncodedInitializer(second);
  builder.MakeEncodedInitializer(duplicate);
  EXPECT_EQ(builder.RemoveDuplicateInitializers(), 1u);
  ASSERT_EQ(builder.ExtendedInitializers().size(), 2u);
  EXPECT_FALSE(builder.Shapes().HasEncodedValue("duplicate"));
  EXPECT_EQ(builder.GetShape("half_scale").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(builder.Shapes().GetEncodedLayout("half_scale").element_bits, 8u);
  EXPECT_EQ(builder.Shapes().GetEncodedLayout("half_scale").record_count, 4u);
  builder.MakeOutput("half_scale");
  builder.MakeOutput("unit_scale");
  const auto exported = builder.ToModel();
  ASSERT_EQ(exported.graph().extended_initializer().size(), 2u);
  EXPECT_EQ(exported.graph().extended_initializer()[0].encoded_value().SerializeAsString(),
            first.SerializeAsString());
  EXPECT_EQ(exported.graph().extended_initializer()[1].encoded_value().SerializeAsString(),
            second.SerializeAsString());
  EXPECT_THROW(builder.ToStandardModel(), core::builder::BuilderError);
  SerializeOptions options;
  options.format = SerializeFormat::kOrtFlatbuffers;
  std::string bytes;
  EXPECT_THROW(exported.SerializeToString(bytes, options), std::invalid_argument);
}

} // namespace Test
