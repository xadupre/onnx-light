// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/runtime_session.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::KernelBase;
using core::runtime::NodeKernelFn;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeSession;
using core::runtime::Tensor;

namespace {

struct KernelRecord {
  const KernelBase *instance;
  const NodeProto *node;
  const float *parameter;
  float scale;
  int runs = 0;
  bool destroyed = false;
  std::vector<size_t> workspace_sizes;
};

struct LifecycleState {
  int factory_calls = 0;
  int attribute_reads = 0;
  std::vector<KernelRecord> kernels;
};

class InstrumentedKernel final : public KernelBase {
public:
  InstrumentedKernel(const NodeProto &node, std::shared_ptr<LifecycleState> state, float scale)
      : KernelBase(core::runtime::KernelContext{}), state_(std::move(state)), scale_(scale),
        id_(state_->kernels.size()) {
    set_node(node);
    state_->kernels.push_back({this, &node, &scale_, scale_, 0, false, {}});
  }

  ~InstrumentedKernel() override { state_->kernels[id_].destroyed = true; }

  void Run(RuntimeContext &rt) override {
    KernelRecord &record = state_->kernels[id_];
    EXPECT_EQ(record.instance, this);
    EXPECT_EQ(record.node, node_);
    EXPECT_EQ(record.parameter, &scale_);
    EXPECT_FLOAT_EQ(record.scale, scale_);
    ++record.runs;
    const Tensor &input = rt.Get(node_->input(0));
    if (input.shape.size() != 2 || input.shape[1] != 20) {
      throw std::invalid_argument("InstrumentedKernel requires shape [N, 20]");
    }
    workspace_.resize(static_cast<size_t>(input.element_count()));
    record.workspace_sizes.push_back(workspace_.size());
    for (size_t i = 0; i < workspace_.size(); ++i) {
      workspace_[i] = input.AsFloat()[i] * scale_;
    }
    rt.Set(node_->output(0),
           Tensor::FromFloat(node_->output(0), input.shape, workspace_, rt.allocator()));
  }

private:
  std::shared_ptr<LifecycleState> state_;
  const float scale_;
  const size_t id_;
  std::vector<float> workspace_;
};

NodeKernelFn MakeFactory(const std::shared_ptr<LifecycleState> &state, float multiplier = 1) {
  // Global dispatch entries outlive tests; they must not retain test state.
  return [weak = std::weak_ptr<LifecycleState>(state), multiplier](const NodeProto &node,
                                                                   RuntimeContext &) {
    auto current = weak.lock();
    if (!current) {
      throw std::logic_error("Lifecycle test registration is no longer active");
    }
    ++current->factory_calls;
    ++current->attribute_reads;
    return std::make_unique<InstrumentedKernel>(node, current, node.attribute(0).f() * multiplier);
  };
}

NodeProto MakeScaleNode(const std::string &domain, const std::string &output, float scale = 2) {
  NodeProto node;
  node.set_domain(domain);
  node.set_op_type("Scale");
  node.add_input("x");
  node.add_output(output);
  auto *attr = node.add_attribute();
  attr->set_name("scale");
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(scale);
  return node;
}

GraphProto MakeGraph(const std::string &domain, bool two_nodes = false) {
  GraphProto graph;
  graph.add_input()->set_name("x");
  graph.add_output()->set_name("y");
  graph.ref_node().push_back(MakeScaleNode(domain, "y"));
  if (two_nodes) {
    graph.add_output()->set_name("z");
    graph.ref_node().push_back(MakeScaleNode(domain, "z", 3));
  }
  return graph;
}

Tensor MakeInput(int64_t rows, float value = 4, int64_t columns = 20) {
  return Tensor::FromFloat("x", {rows, columns},
                           std::vector<float>(static_cast<size_t>(rows * columns), value));
}

void Feed(RuntimeContext &rt, int64_t rows, float value = 4, int64_t columns = 20) {
  rt.Remove("y");
  rt.Remove("z");
  rt.Put("x", MakeInput(rows, value, columns));
}

void ExpectOutput(const Tensor &output, int64_t rows, float value) {
  ASSERT_EQ(output.shape.size(), 2u);
  EXPECT_EQ(output.shape[0], rows);
  EXPECT_EQ(output.shape[1], 20);
  ASSERT_EQ(output.element_count(), rows * 20);
  for (int64_t i = 0; i < output.element_count(); ++i) {
    EXPECT_FLOAT_EQ(output.AsFloat()[i], value);
  }
}

class KernelLifecycle : public testing::TestWithParam<bool> {
protected:
  std::string Domain() const {
    return std::string("test.onnxlight.lifecycle.") +
           testing::UnitTest::GetInstance()->current_test_info()->name() +
           (GetParam() ? ".local" : ".global");
  }

  bool Register(RuntimeContext &rt, const std::string &domain, NodeKernelFn factory,
                bool overwrite = true) {
    if (GetParam()) {
      return rt.RegisterKernelFn(domain, "Scale", core::symbolic::Device::kCPU, std::move(factory),
                                 overwrite);
    }
    return core::runtime::RegisterKernelFn(domain, "Scale", core::symbolic::Device::kCPU,
                                           std::move(factory), overwrite);
  }
};

TEST_P(KernelLifecycle, RetainsPerNodeStateAcrossShapesAndIndependentSessions) {
  const auto state = std::make_shared<LifecycleState>();
  const GraphProto graph = MakeGraph(Domain(), true);
  {
    RuntimeContext first(core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
    RuntimeContext second(core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
    ASSERT_TRUE(Register(first, Domain(), MakeFactory(state)));
    ASSERT_TRUE(Register(second, Domain(), MakeFactory(state)));
    RuntimeSession first_session(first.GetExecutionPlan(graph));
    EXPECT_EQ(state->factory_calls, 0);
    for (int64_t rows : {1, 100, 10}) {
      Feed(first, rows);
      first_session.Run(first);
      ExpectOutput(first.Get("y"), rows, 8);
      ExpectOutput(first.Get("z"), rows, 12);
      EXPECT_EQ(state->factory_calls, 2);
      EXPECT_EQ(state->attribute_reads, 2);
    }
    {
      RuntimeSession second_session(second.GetExecutionPlan(graph));
      Feed(second, 100, 7);
      second_session.Run(second);
      ExpectOutput(second.Get("y"), 100, 14);
      ExpectOutput(second.Get("z"), 100, 21);
      ASSERT_EQ(state->kernels.size(), 4u);
      for (size_t i = 0; i < state->kernels.size(); ++i) {
        EXPECT_EQ(state->kernels[i].node, &graph.node(i % 2));
        EXPECT_FALSE(state->kernels[i].destroyed);
        for (size_t j = 0; j < i; ++j) {
          EXPECT_NE(state->kernels[i].instance, state->kernels[j].instance);
          EXPECT_NE(state->kernels[i].parameter, state->kernels[j].parameter);
        }
      }
      Feed(first, 1, 5);
      first_session.Run(first);
      ExpectOutput(first.Get("y"), 1, 10);
      EXPECT_EQ(state->kernels[0].runs, 4);
      EXPECT_EQ(state->kernels[2].runs, 1);
    }
    EXPECT_FALSE(state->kernels[0].destroyed);
    EXPECT_FALSE(state->kernels[1].destroyed);
    EXPECT_TRUE(state->kernels[2].destroyed);
    EXPECT_TRUE(state->kernels[3].destroyed);
    EXPECT_EQ(state->kernels[0].workspace_sizes, (std::vector<size_t>{20, 2000, 200, 20}));
    EXPECT_EQ(state->kernels[1].workspace_sizes, (std::vector<size_t>{20, 2000, 200, 20}));
    EXPECT_EQ(state->attribute_reads, 4);
  }
  for (const auto &record : state->kernels) {
    EXPECT_TRUE(record.destroyed);
  }
}

TEST_P(KernelLifecycle, ReplacementOnlyAffectsFutureResolutions) {
  const auto original = std::make_shared<LifecycleState>();
  const auto replacement = std::make_shared<LifecycleState>();
  const GraphProto graph = MakeGraph(Domain());
  RuntimeContext rt(core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
  ASSERT_TRUE(Register(rt, Domain(), MakeFactory(original)));
  RuntimeSession existing(rt.GetExecutionPlan(graph));
  Feed(rt, 1);
  existing.Run(rt);
  ASSERT_FALSE(Register(rt, Domain(), MakeFactory(replacement, 10), false));
  {
    RuntimeSession unchanged(rt.GetExecutionPlan(graph));
    Feed(rt, 100);
    unchanged.Run(rt);
    ExpectOutput(rt.Get("y"), 100, 8);
    EXPECT_EQ(original->factory_calls, 2);
    EXPECT_EQ(replacement->factory_calls, 0);
  }
  ASSERT_TRUE(Register(rt, Domain(), MakeFactory(replacement, 10)));
  Feed(rt, 10);
  existing.Run(rt);
  ExpectOutput(rt.Get("y"), 10, 8);
  EXPECT_EQ(original->factory_calls, 2);
  EXPECT_EQ(replacement->factory_calls, 0);
  RuntimeSession future(rt.GetExecutionPlan(graph));
  Feed(rt, 1);
  future.Run(rt);
  ExpectOutput(rt.Get("y"), 1, 80);
  EXPECT_EQ(replacement->factory_calls, 1);
  EXPECT_FALSE(original->kernels[0].destroyed);
}

TEST_P(KernelLifecycle, UnsupportedShapeDoesNotReconstructKernel) {
  const auto state = std::make_shared<LifecycleState>();
  const GraphProto graph = MakeGraph(Domain());
  RuntimeContext rt(core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
  ASSERT_TRUE(Register(rt, Domain(), MakeFactory(state)));
  RuntimeSession session(rt.GetExecutionPlan(graph));
  Feed(rt, 1);
  session.Run(rt);
  Feed(rt, 100, 4, 19);
  EXPECT_THROW(session.Run(rt), std::invalid_argument);
  EXPECT_FALSE(rt.Has("y"));
  EXPECT_EQ(state->factory_calls, 1);
  Feed(rt, 10, 5);
  session.Run(rt);
  ExpectOutput(rt.Get("y"), 10, 10);
  ASSERT_EQ(state->kernels.size(), 1u);
  EXPECT_EQ(state->kernels[0].runs, 3);
  EXPECT_EQ(state->kernels[0].workspace_sizes, (std::vector<size_t>{20, 200}));
  EXPECT_EQ(state->attribute_reads, 1);
}

NodeProto MakeIf(const std::string &condition, const GraphProto &then_branch,
                 const GraphProto &else_branch) {
  NodeProto node;
  node.set_op_type("If");
  node.add_input(condition);
  node.add_output("y");
  for (const auto &branch :
       {std::make_pair("then_branch", &then_branch), std::make_pair("else_branch", &else_branch)}) {
    auto *attr = node.add_attribute();
    attr->set_name(branch.first);
    attr->set_type(AttributeProto::AttributeType::GRAPH);
    *attr->add_g() = *branch.second;
  }
  return node;
}

TEST_P(KernelLifecycle, NestedIfInheritsFactoriesAndReusesEachSelectedBranch) {
  const auto state = std::make_shared<LifecycleState>();
  GraphProto then_branch = MakeGraph(Domain());
  then_branch.ref_input().clear();
  GraphProto else_branch;
  else_branch.add_output()->set_name("y");
  else_branch.ref_node().push_back(MakeScaleNode(Domain(), "y", 3));
  GraphProto nested;
  nested.add_output()->set_name("y");
  nested.ref_node().push_back(MakeIf("inner", then_branch, else_branch));
  GraphProto graph;
  graph.add_input()->set_name("x");
  graph.add_input()->set_name("outer");
  graph.add_input()->set_name("inner");
  graph.add_output()->set_name("y");
  graph.ref_node().push_back(MakeIf("outer", nested, else_branch));
  {
    RuntimeContext rt(core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
    ASSERT_TRUE(Register(rt, Domain(), MakeFactory(state)));
    rt.Set("outer", Tensor::FromBool("outer", {}, {true}));
    RuntimeSession session(rt.GetExecutionPlan(graph));
    int index = 0;
    for (int64_t rows : {1, 100, 10}) {
      Feed(rt, rows);
      const bool take_then = index != 1;
      rt.Put("inner", Tensor::FromBool("inner", {}, {static_cast<uint8_t>(take_then)}));
      session.Run(rt);
      ExpectOutput(rt.Get("y"), rows, take_then ? 8 : 12);
      EXPECT_EQ(state->factory_calls, index == 0 ? 1 : 2);
      ++index;
    }
    ASSERT_EQ(state->kernels.size(), 2u);
    EXPECT_EQ(state->kernels[0].runs, 2);
    EXPECT_EQ(state->kernels[1].runs, 1);
    EXPECT_EQ(state->kernels[0].workspace_sizes, (std::vector<size_t>{20, 200}));
  }
  for (const auto &record : state->kernels) {
    EXPECT_TRUE(record.destroyed);
  }
}

TEST_P(KernelLifecycle, NestedFunctionsInheritFactoriesAndIsolateCallSites) {
  const auto state = std::make_shared<LifecycleState>();
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(18);
  auto *inner = model.add_functions();
  inner->set_domain(Domain());
  inner->set_name("Inner");
  inner->add_input("x");
  inner->add_output("y");
  inner->ref_node().push_back(MakeScaleNode(Domain(), "y"));
  auto *outer = model.add_functions();
  outer->set_domain(Domain());
  outer->set_name("Outer");
  outer->add_input("x");
  outer->add_output("y");
  NodeProto call;
  call.set_domain(Domain());
  call.set_op_type("Inner");
  call.add_input("x");
  call.add_output("y");
  outer->ref_node().push_back(call);
  auto *graph = model.add_graph();
  graph->add_input()->set_name("x");
  for (const std::string output : {"y", "z"}) {
    graph->add_output()->set_name(output);
    NodeProto outer_call;
    outer_call.set_domain(Domain());
    outer_call.set_op_type("Outer");
    outer_call.add_input("x");
    outer_call.add_output(output);
    graph->ref_node().push_back(outer_call);
  }
  {
    RuntimeContext rt(core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
    ASSERT_TRUE(Register(rt, Domain(), MakeFactory(state)));
    core::runtime::RegisterModelFunctions(model, rt);
    RuntimeSession session(rt.GetExecutionPlan(*graph));
    for (int64_t rows : {1, 100, 10}) {
      Feed(rt, rows);
      session.Run(rt);
      ExpectOutput(rt.Get("y"), rows, 8);
      ExpectOutput(rt.Get("z"), rows, 8);
      EXPECT_EQ(state->factory_calls, 2);
    }
    ASSERT_EQ(state->kernels.size(), 2u);
    EXPECT_NE(state->kernels[0].instance, state->kernels[1].instance);
    EXPECT_NE(state->kernels[0].parameter, state->kernels[1].parameter);
    for (const auto &record : state->kernels) {
      EXPECT_EQ(record.runs, 3);
      EXPECT_EQ(record.workspace_sizes, (std::vector<size_t>{20, 2000, 200}));
    }
    EXPECT_EQ(state->attribute_reads, 2);
  }
  for (const auto &record : state->kernels) {
    EXPECT_TRUE(record.destroyed);
  }
}

class GlobalCallbackGuard {
public:
  explicit GlobalCallbackGuard(std::string domain) : domain_(std::move(domain)) {}
  ~GlobalCallbackGuard() { core::runtime::UnregisterGlobalCustomKernel(domain_, "Scale"); }

private:
  std::string domain_;
};

TEST_P(KernelLifecycle, CallbackKeepsOriginalNodeAndIndependentMutableCapture) {
  GraphProto graph = MakeGraph(Domain(), true);
  for (auto &node : graph.ref_node()) {
    auto *attr = node.add_attribute();
    attr->set_name("heavy");
    attr->set_type(AttributeProto::AttributeType::STRING);
    attr->set_s(std::string(128 * 1024, 'k'));
  }
  const std::vector<const NodeProto *> nodes{&graph.node(0), &graph.node(1)};
  const std::vector<const char *> payloads{graph.node(0).attribute(1).s().data(),
                                           graph.node(1).attribute(1).s().data()};
  auto callback = [nodes, payloads, calls = 0](const NodeProto &node, RuntimeContext &rt) mutable {
    const size_t index = node.output(0) == "y" ? 0 : 1;
    EXPECT_EQ(&node, nodes[index]);
    EXPECT_EQ(&node.attribute(1), &nodes[index]->attribute(1));
    EXPECT_EQ(node.attribute(1).s().data(), payloads[index]);
    EXPECT_EQ(node.attribute(1).s().size(), 128 * 1024u);
    ++calls;
    const auto &input = rt.Get(node.input(0));
    rt.Set(node.output(0),
           Tensor::FromFloat(node.output(0), input.shape,
                             std::vector<float>(static_cast<size_t>(input.element_count()),
                                                static_cast<float>(calls)),
                             rt.allocator()));
  };
  GlobalCallbackGuard cleanup(Domain());
  RuntimeContext first(core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
  RuntimeContext second(core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
  if (GetParam()) {
    first.RegisterCustomKernel(Domain(), "Scale", callback);
    second.RegisterCustomKernel(Domain(), "Scale", callback);
  } else {
    core::runtime::RegisterGlobalCustomKernel(Domain(), "Scale", callback);
  }
  RuntimeSession first_session(first.GetExecutionPlan(graph));
  RuntimeSession second_session(second.GetExecutionPlan(graph));
  int calls = 0;
  for (int64_t rows : {1, 100, 10}) {
    Feed(first, rows);
    first_session.Run(first);
    ++calls;
    ExpectOutput(first.Get("y"), rows, static_cast<float>(calls));
    ExpectOutput(first.Get("z"), rows, static_cast<float>(calls));
  }
  Feed(second, 100);
  second_session.Run(second);
  ExpectOutput(second.Get("y"), 100, 1);
  ExpectOutput(second.Get("z"), 100, 1);
  if (GetParam()) {
    ASSERT_TRUE(first.UnregisterCustomKernel(Domain(), "Scale"));
  } else {
    ASSERT_TRUE(core::runtime::UnregisterGlobalCustomKernel(Domain(), "Scale"));
  }
  Feed(first, 1);
  first_session.Run(first);
  ExpectOutput(first.Get("y"), 1, 4);
  ExpectOutput(first.Get("z"), 1, 4);
  RuntimeSession unresolved(first.GetExecutionPlan(graph));
  Feed(first, 1);
  EXPECT_THROW(unresolved.Run(first), std::invalid_argument);
}

TEST(KernelLifecyclePrecedence, LocalFactoryOverridesGlobalCallbackAndDispatchFactory) {
  const std::string domain = "test.onnxlight.lifecycle.precedence";
  const auto global = std::make_shared<LifecycleState>();
  const auto local = std::make_shared<LifecycleState>();
  const GraphProto graph = MakeGraph(domain);
  GlobalCallbackGuard cleanup(domain);
  ASSERT_TRUE(core::runtime::RegisterKernelFn(domain, "Scale", core::symbolic::Device::kCPU,
                                              MakeFactory(global)));
  core::runtime::RegisterGlobalCustomKernel(
      domain, "Scale", [](const NodeProto &, RuntimeContext &) {
        ADD_FAILURE() << "A context-local factory must take precedence over global callbacks";
      });
  RuntimeContext rt(core::runtime::KernelContext(core::backend_test::DefaultOpset(18)));
  ASSERT_TRUE(
      rt.RegisterKernelFn(domain, "Scale", core::symbolic::Device::kCPU, MakeFactory(local, 5)));
  RuntimeSession session(rt.GetExecutionPlan(graph));
  Feed(rt, 1);
  session.Run(rt);
  ExpectOutput(rt.Get("y"), 1, 40);
  ASSERT_TRUE(core::runtime::UnregisterGlobalCustomKernel(domain, "Scale"));
  Feed(rt, 100);
  session.Run(rt);
  ExpectOutput(rt.Get("y"), 100, 40);
  RuntimeSession fresh(rt.GetExecutionPlan(graph));
  Feed(rt, 10);
  fresh.Run(rt);
  ExpectOutput(rt.Get("y"), 10, 40);
  EXPECT_EQ(global->factory_calls, 0);
  EXPECT_EQ(local->factory_calls, 2);
}

INSTANTIATE_TEST_SUITE_P(GlobalAndLocal, KernelLifecycle, testing::Bool(),
                         [](const testing::TestParamInfo<bool> &info) {
                           return info.param ? "Local" : "Global";
                         });

} // namespace
