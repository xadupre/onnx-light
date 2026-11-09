// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/patterns/canonicalization/identity_pattern.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

core::builder::GraphBuilder::SchemaLookupFn CollisionSchemaLookup() {
  onnx_shapes::RegisterShapeFunctions();
  onnx_kernels::RegisterKernelFunctions();
  return [](const std::string &op) { return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false); };
}

core::builder::OperatorRegistry CollisionRegistry() {
  core::builder::OperatorRegistry registry;
  for (const std::string name : {"FastGelu", "SkipSimplifiedLayerNormalization"})
    registry.RegisterShapeInference("com.microsoft", name,
                                    [](core::shapes::ShapesContext &ctx, const NodeProto &node) {
                                      ctx.CopyValueFrom(node.output(0), ctx, node.input(0));
                                    });
  return registry;
}

ModelProto CollisionModel(const std::string &domain, const std::string &name,
                          const std::string &overload) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset("", 18);
  if (NormaliseDomain(domain) != kDefaultOnnxDomain)
    model.add_opset(domain, 1);
  auto &function = *model.add_functions();
  function.set_domain(domain);
  function.set_name(name);
  function.set_overload(overload);
  function.add_opset("", 18);
  function.add_input("a");
  function.add_output("b");
  function.add_attribute("required_only_by_local_function");
  function.add_node(MakeNode("Neg", {"a"}, {"b"}));
  auto &graph = *model.mutable_graph();
  graph.set_name("collision");
  const auto value_info = [](const std::string &value, std::initializer_list<int64_t> shape) {
    ValueInfoProto info;
    info.set_name(value);
    auto &tensor = *info.mutable_type()->mutable_tensor_type();
    tensor.set_elem_type(TensorProto::FLOAT);
    for (auto dim : shape)
      tensor.mutable_shape()->add_dim()->set_dim_value(dim);
    return info;
  };
  *graph.add_input() = value_info("x", {1, 2});
  *graph.add_output() = value_info("y", {1, 2});
  auto node = MakeNode(name.c_str(), {"x"}, {"y"}, domain.c_str());
  if (name == "SkipSimplifiedLayerNormalization") {
    *graph.add_input() = value_info("skip", {1, 2});
    *graph.add_input() = value_info("scale", {2});
    node.add_input("skip");
    node.add_input("scale");
  }
  node.set_overload(overload);
  graph.add_node(std::move(node));
  return model;
}

RewritingProto CollisionPlan(const ModelProto &model, bool duplicate) {
  RewritingProto plan;
  auto replacement = model.functions(0);
  // An ignored declaration must not be normalized or used as a fallback.
  replacement.ref_node()[0].set_op_type(duplicate ? "UnavailableCollisionBody" : "Abs");
  plan.ref_added_functions().push_back(std::move(replacement));
  if (duplicate) {
    plan.ref_added_functions().push_back(model.functions(0));
  }
  return plan;
}

class IntroduceFastGeluPattern : public core::builder::PatternOptimization {
public:
  IntroduceFastGeluPattern() : PatternOptimization(0, "IntroduceFastGelu") {}

  core::builder::MatchResult Match(core::builder::GraphGraph &,
                                   const NodeProto &candidate) const override {
    if (candidate.op_type() == "Relu" && NormaliseDomain(candidate.domain()) == kDefaultOnnxDomain)
      return {this, {&candidate}};
    return {};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &graph,
        const std::vector<const NodeProto *> &nodes) const override {
    graph.Builder().SetOpsetVersion("com.microsoft", 1);
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(MakeNode("FastGelu", {nodes[0]->input(0).value()},
                                    {nodes[0]->output(0).value()}, "com.microsoft"));
    return replacements;
  }
};

void SetOptimizableBody(ModelProto &model) {
  auto transpose = MakeNode("Transpose", {"a"}, {"b"});
  auto &perm = *transpose.add_attribute();
  perm.set_name("perm");
  perm.set_type(AttributeProto::INTS);
  perm.add_ints(0);
  perm.add_ints(1);
  model.ref_functions()[0].ref_node()[0] = std::move(transpose);
  auto info = model.graph().input(0);
  info.set_name("a");
  model.ref_functions()[0].ref_value_info().push_back(std::move(info));
}

ModelProto VersionedCallerModel(bool nested, bool default_attribute) {
  auto model = CollisionModel("custom", "F", "variant");
  model.ref_opset_import()[1].set_version(2);
  model.ref_functions()[0].ref_attribute().clear();
  auto &wrapper = *model.add_functions();
  wrapper.set_domain("local.control");
  wrapper.set_name("Wrapper");
  wrapper.add_opset("", 18);
  wrapper.add_opset("custom", 1);
  wrapper.add_input("a");
  wrapper.add_output("b");
  auto call = MakeNode("F", {"a"}, {"b"}, "custom");
  call.set_overload("variant");
  if (nested || default_attribute) {
    auto identity = MakeNode("Identity", {"a"}, {"b"});
    auto &attribute =
        default_attribute ? *wrapper.add_attribute_proto() : *identity.add_attribute();
    attribute.set_name("body");
    attribute.set_type(AttributeProto::GRAPH);
    attribute.mutable_g()->set_name("nested");
    attribute.mutable_g()->add_node(std::move(call));
    wrapper.add_node(std::move(identity));
  } else {
    wrapper.add_node(std::move(call));
  }
  // The root selects the registered operator; only the wrapper selects local F.
  return model;
}

core::builder::OperatorRegistry VersionedRegistry() {
  core::builder::OperatorRegistry registry;
  registry.RegisterSchema(core::schema::LightOpSchema(
      "F", "custom", 2, "", {{"x", "", "tensor(float)"}}, {{"y", "", "tensor(float)"}}, {}));
  return registry;
}

} // namespace

TEST(RewritingOperatorPrecedence, CapturesAndReplaysRegisteredOperators) {
  for (const std::string domain : {"", "ai.onnx", "onnx.ai", "com.microsoft"}) {
    const std::vector<std::string> names =
        domain == "com.microsoft"
            ? std::vector<std::string>{"FastGelu", "SkipSimplifiedLayerNormalization"}
            : std::vector<std::string>{"Identity"};
    for (const auto &name : names) {
      for (const std::string overload : {"", "variant:one%"}) {
        for (bool duplicate : {false, true}) {
          SCOPED_TRACE(domain + "/" + name + "/" + overload + "/" + std::to_string(duplicate));
          auto model = CollisionModel(domain, name, overload);
          if (domain == "onnx.ai")
            model.mutable_graph()->ref_node()[0].set_domain("ai.onnx");
          const auto original = model.SerializeAsString();
          core::builder::GraphBuilder source(model, CollisionSchemaLookup(), CollisionRegistry());
          const auto source_before = source.ToModel().SerializeAsString();
          const auto registrations = core::runtime::KernelDispatchTable().size();
          const auto shape_registrations = core::shapes::DispatchTable().size();
          const auto plan = source.CaptureRewritings(CollisionPlan(model, duplicate));
          RewritingProto saved;
          ASSERT_TRUE(saved.ParseFromString(plan.SerializeAsString()));
          auto result = source.ApplyRewritings(saved);
          const auto replayed = result.ToModel();
          EXPECT_EQ(replayed.functions(0).node(0).op_type(), "Neg");
          EXPECT_TRUE(result.IsRegisteredOperator(replayed.graph().node(0)));
          EXPECT_EQ(result.GetShape("y").Shape().Rank(), 2u);
          EXPECT_EQ(result.GetShape("y").Shape()[1].AsInt(), 2);
          // Contrib shape callbacks are builder-local; Identity also exercises native execution.
          if (domain != "com.microsoft") {
            core::runtime::RuntimeContext context(
                core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
            FunctionProto sentinel;
            context.functions().emplace("unrelated:Sentinel:", &sentinel);
            context.Set("x", core::runtime::Tensor::FromFloat("x", {1, 2}, {2, 2}));
            core::runtime::RuntimeSession session(replayed);
            ASSERT_NO_THROW(session.Run(context));
            EXPECT_FLOAT_EQ(context.Get("y").AsFloat()[0], 2);
            EXPECT_EQ(context.functions().size(), 1u);
            EXPECT_EQ(context.functions().at("unrelated:Sentinel:"), &sentinel);
          }
          EXPECT_EQ(model.SerializeAsString(), original);
          EXPECT_EQ(source.ToModel().SerializeAsString(), source_before);
          EXPECT_EQ(core::runtime::KernelDispatchTable().size(), registrations);
          EXPECT_EQ(core::shapes::DispatchTable().size(), shape_registrations);
        }
      }
    }
  }
}

TEST(RewritingOperatorPrecedence, FunctionOnlyResolutionAndConflictsRemain) {
  for (const std::string overload : {"", "variant:one%"}) {
    auto model = CollisionModel("local.control", "FastGelu", overload);
    model.ref_functions()[0].ref_attribute().clear();
    core::builder::GraphBuilder source(model, CollisionSchemaLookup(), CollisionRegistry());
    const auto before = source.ToModel().SerializeAsString();
    RewritingProto unchanged;
    unchanged.ref_added_functions().push_back(model.functions(0));
    const auto replayed = source.ApplyRewritings(source.CaptureRewritings(unchanged)).ToModel();
    core::runtime::RuntimeContext context(
        core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
    context.Set("x", core::runtime::Tensor::FromFloat("x", {1, 2}, {2, 2}));
    core::runtime::RuntimeSession session(replayed);
    session.Run(context);
    EXPECT_FLOAT_EQ(context.Get("y").AsFloat()[0], -2);
    for (bool duplicate : {false, true}) {
      auto conflict = unchanged;
      conflict.ref_added_functions()[0].ref_node()[0].set_op_type("Abs");
      if (duplicate)
        conflict.ref_added_functions().push_back(model.functions(0));
      EXPECT_THROW(source.CaptureRewritings(conflict), core::builder::BuilderError);
      auto saved = source.CaptureRewritings(unchanged);
      saved.ref_added_functions() = conflict.added_functions();
      try {
        source.ApplyRewritings(saved);
        FAIL() << "Expected genuinely conflicting local functions to be rejected.";
      } catch (const core::builder::BuilderError &error) {
        EXPECT_NE(std::string(error.what()).find("conflicting"), std::string::npos);
      }
    }
    EXPECT_EQ(source.ToModel().SerializeAsString(), before);
  }
}

TEST(RewritingOperatorPrecedence, UsesRequiredImportsForExistingAndNewDeclarations) {
  for (bool existing : {false, true}) {
    SCOPED_TRACE(existing);
    auto model = CollisionModel("com.microsoft", "FastGelu", "variant");
    auto edits = CollisionPlan(model, false);
    edits.ref_added_functions().push_back(model.functions(0));
    auto &opset = *edits.add_required_opset_imports();
    opset.set_domain("com.microsoft");
    opset.set_version(1);
    model.ref_opset_import().erase(model.ref_opset_import().begin() + 1);
    model.mutable_graph()->ref_node()[0] = MakeNode("Identity", {"x"}, {"y"});
    auto &rewrite = *edits.add_rewrites();
    rewrite.add_matched_nodes(0);
    auto call = MakeNode("FastGelu", {"x"}, {"y"}, "com.microsoft");
    call.set_overload("variant");
    rewrite.ref_added_nodes().push_back(std::move(call));
    rewrite.add_added_nodes_positions(0);
    if (!existing)
      model.clr_functions();
    core::builder::GraphBuilder source(model, CollisionSchemaLookup(), CollisionRegistry());
    const auto before = source.ToModel().SerializeAsString();
    auto result = source.ApplyRewritings(source.CaptureRewritings(edits));
    EXPECT_EQ(result.OpsetVersion("com.microsoft"), 1);
    const auto replayed = result.ToModel();
    ASSERT_EQ(replayed.functions().size(), 1u);
    EXPECT_EQ(replayed.functions(0).node(0).op_type(), existing ? "Neg" : "Abs");
    EXPECT_TRUE(result.IsRegisteredOperator(result.Nodes()[0]));
    EXPECT_EQ(source.ToModel().SerializeAsString(), before);
  }
}

TEST(RewritingOperatorPrecedence, SelectedOperatorFailureDoesNotUseFunction) {
  auto model = CollisionModel("", "Identity", "");
  model.mutable_graph()->ref_node()[0].ref_input().clear();
  core::builder::GraphBuilder source(CollisionModel("", "Identity", ""), CollisionSchemaLookup());
  const auto before = source.ToModel().SerializeAsString();
  RewritingProto edits = CollisionPlan(model, false);
  auto &rewrite = *edits.add_rewrites();
  rewrite.add_matched_nodes(0);
  rewrite.ref_added_nodes().push_back(model.graph().node(0));
  rewrite.add_added_nodes_positions(0);
  auto valid = edits;
  valid.ref_rewrites()[0].ref_added_nodes()[0].add_input("x");
  auto saved = source.CaptureRewritings(valid);
  saved.ref_rewrites()[0].ref_added_nodes()[0].ref_input().clear();
  for (bool replay : {false, true}) {
    SCOPED_TRACE(replay);
    try {
      if (replay)
        source.ApplyRewritings(saved);
      else
        source.CaptureRewritings(edits);
      FAIL() << "Expected the selected Identity operator to reject missing inputs.";
    } catch (const std::invalid_argument &error) {
      const std::string message = error.what();
      EXPECT_NE(message.find("Identity"), std::string::npos);
      EXPECT_EQ(message.find("required_only_by_local_function"), std::string::npos);
      EXPECT_EQ(message.find("conflicting function"), std::string::npos);
    }
  }
  EXPECT_EQ(source.ToModel().SerializeAsString(), before);
}

TEST(RewritingOperatorPrecedence, OptimizedPlansKeepIgnoredBodiesReplayable) {
  auto model = CollisionModel("com.microsoft", "FastGelu", "");
  auto transpose = MakeNode("Transpose", {"a"}, {"b"});
  auto &perm = *transpose.add_attribute();
  perm.set_name("perm");
  perm.set_type(AttributeProto::INTS);
  perm.add_ints(0);
  perm.add_ints(1);
  model.ref_functions()[0].ref_node()[0] = transpose;
  auto info = model.graph().input(0);
  info.set_name("a");
  model.ref_functions()[0].ref_value_info().push_back(info);
  transpose.ref_input()[0] = "x";
  transpose.ref_output()[0] = "transposed";
  model.mutable_graph()->ref_node()[0].ref_input()[0] = "transposed";
  auto call = model.graph().node(0);
  model.mutable_graph()->ref_node().clear();
  model.mutable_graph()->add_node(std::move(transpose));
  model.mutable_graph()->add_node(std::move(call));
  core::builder::GraphBuilder optimized(model, CollisionSchemaLookup(), CollisionRegistry());
  core::builder::GraphBuilder source(model, CollisionSchemaLookup(), CollisionRegistry());
  const auto before = source.ToModel().SerializeAsString();
  const auto plan =
      core::builder::GraphGraph(optimized, {std::make_shared<onnx_patterns::IdentityPattern>()})
          .Optimize(1, nullptr, {}, false);
  EXPECT_FALSE(plan.rewrites().empty());
  EXPECT_EQ(optimized.ToModel().functions(0).node(0).op_type(), "Transpose");
  RewritingProto saved;
  ASSERT_TRUE(saved.ParseFromString(plan.SerializeAsString()));
  auto replayed = source.ApplyRewritings(saved);
  EXPECT_EQ(replayed.RewritingHash(), optimized.RewritingHash());
  EXPECT_EQ(replayed.ToModel().functions(0).node(0).op_type(), "Transpose");
  EXPECT_EQ(source.ToModel().SerializeAsString(), before);
}

TEST(RewritingOperatorPrecedence, OptimizedPlansIntroduceCollidingImports) {
  auto model = CollisionModel("com.microsoft", "FastGelu", "");
  model.ref_opset_import().erase(model.ref_opset_import().begin() + 1);
  model.ref_functions()[0].ref_attribute().clear();
  SetOptimizableBody(model);
  model.mutable_graph()->ref_node()[0] = MakeNode("Relu", {"x"}, {"y"});
  core::builder::GraphBuilder optimized(model, CollisionSchemaLookup(), CollisionRegistry());
  core::builder::GraphBuilder source(model, CollisionSchemaLookup(), CollisionRegistry());
  const auto before = source.ToModel().SerializeAsString();
  const auto plan =
      core::builder::GraphGraph(optimized, {std::make_shared<IntroduceFastGeluPattern>(),
                                            std::make_shared<onnx_patterns::IdentityPattern>()})
          .Optimize(1, nullptr, {}, false);
  ASSERT_EQ(plan.replaced_function_ids().size(), 1u);
  ASSERT_FALSE(plan.required_opset_imports().empty());
  RewritingProto saved;
  ASSERT_TRUE(saved.ParseFromString(plan.SerializeAsString()));
  auto replayed = source.ApplyRewritings(saved);
  EXPECT_EQ(replayed.RewritingHash(), optimized.RewritingHash());
  EXPECT_EQ(replayed.OpsetVersion("com.microsoft"), 1);
  EXPECT_TRUE(replayed.IsRegisteredOperator(replayed.Nodes()[0]));
  EXPECT_EQ(source.ToModel().SerializeAsString(), before);
}

TEST(RewritingOperatorPrecedence, ExplicitRegisteredReplacementsAreValidated) {
  const auto model = CollisionModel("", "Identity", "variant");
  core::builder::GraphBuilder source(model, CollisionSchemaLookup());
  const auto identity = core::shapes::ShapesContext::LocalFunctionKey("", "Identity", "variant");
  auto replacement = CollisionPlan(model, false);
  replacement.add_replaced_function_ids(identity);
  auto result = source.ApplyRewritings(source.CaptureRewritings(replacement));
  EXPECT_EQ(result.LocalFunction("Identity").Nodes()[0].op_type(), "Abs");
  EXPECT_TRUE(result.IsRegisteredOperator(result.Nodes()[0]));
  auto conflicting = CollisionPlan(model, true);
  conflicting.add_replaced_function_ids(identity);
  EXPECT_THROW(source.CaptureRewritings(conflicting), core::builder::BuilderError);
}

TEST(RewritingOperatorPrecedence, LocalCallerImportsPreserveConflictsAndReplacements) {
  for (const auto mode : {0, 1, 2}) {
    SCOPED_TRACE(mode);
    const auto model = VersionedCallerModel(mode == 1, mode == 2);
    core::builder::GraphBuilder source(model, CollisionSchemaLookup(), VersionedRegistry());
    const auto before = source.ToModel().SerializeAsString();
    EXPECT_TRUE(source.IsRegisteredOperator(source.Nodes()[0]));
    for (bool duplicate : {false, true}) {
      auto edits = CollisionPlan(model, false);
      if (duplicate)
        edits.ref_added_functions().push_back(model.functions(0));
      EXPECT_THROW(source.CaptureRewritings(edits), core::builder::BuilderError);
    }
    auto replacement = CollisionPlan(model, false);
    replacement.add_replaced_function_ids(
        core::shapes::ShapesContext::LocalFunctionKey("custom", "F", "variant"));
    auto result = source.ApplyRewritings(source.CaptureRewritings(replacement));
    EXPECT_EQ(result.LocalFunction("F").Nodes()[0].op_type(), "Abs");
    if (mode == 0) {
      auto runtime_model = result.ToModel();
      runtime_model.add_opset("local.control", 1);
      runtime_model.mutable_graph()->ref_node()[0] =
          MakeNode("Wrapper", {"x"}, {"y"}, "local.control");
      core::runtime::RuntimeContext context(
          core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
      context.Set("x", core::runtime::Tensor::FromFloat("x", {1, 2}, {-2, -2}));
      core::runtime::RuntimeSession session(runtime_model);
      session.Run(context);
      EXPECT_FLOAT_EQ(context.Get("y").AsFloat()[0], 2);
    }
    EXPECT_EQ(source.ToModel().SerializeAsString(), before);
  }
}

TEST(RewritingOperatorPrecedence, AddedCallerImportsPreserveConflicts) {
  auto model = VersionedCallerModel(false, false);
  const auto wrapper = model.functions(1);
  model.ref_functions().erase(model.ref_functions().begin() + 1);
  core::builder::GraphBuilder source(model, CollisionSchemaLookup(), VersionedRegistry());
  auto edits = CollisionPlan(model, false);
  edits.ref_added_functions().push_back(wrapper);
  EXPECT_THROW(source.CaptureRewritings(edits), core::builder::BuilderError);
}

TEST(RewritingOperatorPrecedence, DiscardedProposalsDoNotIntroduceLocalCallers) {
  for (bool reverse : {false, true}) {
    for (bool replace_caller : {false, true}) {
      SCOPED_TRACE(reverse);
      SCOPED_TRACE(replace_caller);
      auto model = CollisionModel("custom", "F", "A");
      model.ref_opset_import()[1].set_version(2);
      model.ref_functions()[0].ref_attribute().clear();
      auto callee = model.functions(0);
      callee.set_overload("B");
      model.ref_functions().push_back(callee);
      core::builder::GraphBuilder source(model, CollisionSchemaLookup(), VersionedRegistry());
      const auto before = source.ToModel().SerializeAsString();
      RewritingProto edits;
      auto caller = model.functions(0);
      caller.add_opset("custom", 1);
      caller.ref_node()[0] = MakeNode("F", {"a"}, {"b"}, "custom");
      caller.ref_node()[0].set_overload("B");
      callee.ref_node()[0].set_op_type("Abs");
      edits.ref_added_functions().push_back(reverse ? callee : caller);
      edits.ref_added_functions().push_back(reverse ? caller : callee);
      if (replace_caller) {
        edits.add_replaced_function_ids(
            core::shapes::ShapesContext::LocalFunctionKey("custom", "F", "A"));
        EXPECT_THROW(source.CaptureRewritings(edits), core::builder::BuilderError);
      } else {
        const auto replayed = source.ApplyRewritings(source.CaptureRewritings(edits));
        EXPECT_EQ(replayed.RewritingHash(), source.RewritingHash());
        EXPECT_EQ(replayed.ToModel().SerializeAsString(), before);
      }
      EXPECT_EQ(source.ToModel().SerializeAsString(), before);
    }
  }
}

TEST(RewritingOperatorPrecedence, ReplacedBodiesDoNotRetainOldLocalCallers) {
  auto model = VersionedCallerModel(false, false);
  core::builder::GraphBuilder source(model, CollisionSchemaLookup(), VersionedRegistry());
  auto edits = CollisionPlan(model, false);
  auto wrapper = model.functions(1);
  wrapper.ref_node()[0] = MakeNode("Identity", {"a"}, {"b"});
  edits.ref_added_functions().push_back(wrapper);
  edits.add_replaced_function_ids(
      core::shapes::ShapesContext::LocalFunctionKey("local.control", "Wrapper", ""));
  const auto replayed = source.ApplyRewritings(source.CaptureRewritings(edits));
  EXPECT_EQ(replayed.LocalFunction("F").Nodes()[0].op_type(), "Neg");
  EXPECT_EQ(replayed.LocalFunction("Wrapper").Nodes()[0].op_type(), "Identity");
}

TEST(RewritingOperatorPrecedence, OptimizesBodiesSelectedByLocalCallerImports) {
  auto model = VersionedCallerModel(false, false);
  SetOptimizableBody(model);
  core::builder::GraphBuilder optimized(model, CollisionSchemaLookup(), VersionedRegistry());
  core::builder::GraphBuilder source(model, CollisionSchemaLookup(), VersionedRegistry());
  const auto plan =
      core::builder::GraphGraph(optimized, {std::make_shared<onnx_patterns::IdentityPattern>()})
          .Optimize(1, nullptr, {}, false);
  ASSERT_FALSE(plan.replaced_function_ids().empty());
  auto replayed = source.ApplyRewritings(plan);
  EXPECT_EQ(replayed.RewritingHash(), optimized.RewritingHash());
  ASSERT_EQ(replayed.LocalFunction("F").Nodes().size(), 1u);
  EXPECT_EQ(replayed.LocalFunction("F").Nodes()[0].op_type(), "Identity");
}
