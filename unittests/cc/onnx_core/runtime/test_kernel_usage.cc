// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::KernelBase;
using core::runtime::KernelContext;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeSession;
using core::runtime::Tensor;

TEST(KernelUsage, DisabledByDefaultAndIndependentContexts) {
  RuntimeContext first;
  RuntimeContext second;
  EXPECT_FALSE(first.kernel_usage_enabled());
  first.RecordKernelUsage("ignored");
  EXPECT_TRUE(first.GetKernelUsage().empty());

  first.set_kernel_usage_enabled(true);
  second.set_kernel_usage_enabled(true);
  first.RecordKernelUsage("first");
  second.RecordKernelUsage("second");
  auto snapshot = first.GetKernelUsage();
  snapshot.push_back("snapshot-only");
  EXPECT_EQ(first.GetKernelUsage(), (std::vector<std::string>{"first"}));

  first.set_kernel_usage_enabled(false);
  first.RecordKernelUsage("ignored");
  EXPECT_EQ(first.GetKernelUsage(), (std::vector<std::string>{"first"}));
  first.ClearKernelUsage();
  EXPECT_TRUE(first.GetKernelUsage().empty());
  EXPECT_TRUE(second.kernel_usage_enabled());
  EXPECT_EQ(second.GetKernelUsage(), (std::vector<std::string>{"second"}));
  EXPECT_TRUE(first.custom_kernels().empty());
  EXPECT_TRUE(second.custom_kernels().empty());
}

TEST(KernelUsage, BoundedLogOwnsNamesAndCanBeReused) {
  RuntimeContext rt;
  rt.set_kernel_usage_enabled(true);
  std::string name = "owned";
  rt.RecordKernelUsage(name);
  name[0] = 'X';
  for (size_t i = 1; i <= RuntimeContext::kKernelUsageLimit; ++i) {
    rt.RecordKernelUsage("repeated");
  }
  const auto snapshot = rt.GetKernelUsage();
  ASSERT_EQ(snapshot.size(), RuntimeContext::kKernelUsageLimit);
  EXPECT_EQ(snapshot.front(), "owned");
  EXPECT_EQ(snapshot.back(), "repeated");
  rt.ClearKernelUsage();
  EXPECT_TRUE(rt.kernel_usage_enabled());
  rt.RecordKernelUsage("new");
  EXPECT_EQ(rt.GetKernelUsage(), (std::vector<std::string>{"new"}));
  EXPECT_EQ(snapshot.size(), RuntimeContext::kKernelUsageLimit);
}

TEST(KernelUsage, ChildrenShareStateEvenWhenCreatedBeforeEnabling) {
  RuntimeContext parent;
  RuntimeContext subgraph = parent.MakeSubgraphContext("body");
  RuntimeContext function = subgraph.MakeFunctionContext();
  RuntimeContext nested = function.MakeSubgraphContext("then_branch");
  parent.set_kernel_usage_enabled(true);
  subgraph.RecordKernelUsage("subgraph");
  function.RecordKernelUsage("function");
  nested.RecordKernelUsage("nested");
  EXPECT_EQ(parent.GetKernelUsage(), (std::vector<std::string>{"subgraph", "function", "nested"}));
  function.ClearKernelUsage();
  EXPECT_TRUE(parent.GetKernelUsage().empty());
  nested.set_kernel_usage_enabled(false);
  parent.RecordKernelUsage("ignored");
  EXPECT_FALSE(subgraph.kernel_usage_enabled());
  EXPECT_TRUE(parent.GetKernelUsage().empty());
  EXPECT_TRUE(function.custom_kernels().empty());
}

TEST(KernelUsage, ChildKeepsStateAliveAfterParentDestruction) {
  RuntimeContext child = [] {
    RuntimeContext parent;
    parent.set_kernel_usage_enabled(true);
    parent.RecordKernelUsage("parent");
    return parent.MakeFunctionContext();
  }();
  child.RecordKernelUsage("child");
  child.Clear();
  EXPECT_TRUE(child.kernel_usage_enabled());
  EXPECT_EQ(child.GetKernelUsage(), (std::vector<std::string>{"parent", "child"}));
}

TEST(KernelUsage, ConcurrentContextsDoNotMixNamesOrLoseAppends) {
  RuntimeContext first;
  RuntimeContext second;
  first.set_kernel_usage_enabled(true);
  second.set_kernel_usage_enabled(true);
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i) {
    workers.emplace_back([&] {
      for (int j = 0; j < 100; ++j) {
        first.RecordKernelUsage("first");
        second.RecordKernelUsage("second");
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_EQ(first.GetKernelUsage(), std::vector<std::string>(400, "first"));
  EXPECT_EQ(second.GetKernelUsage(), std::vector<std::string>(400, "second"));
}

TEST(KernelUsage, ConcurrentRecordingSnapshotClearAndDisable) {
  RuntimeContext rt;
  RuntimeContext child = rt.MakeFunctionContext();
  rt.set_kernel_usage_enabled(true);
  std::atomic<bool> start{false};
  std::thread writer([&] {
    while (!start.load()) {
      std::this_thread::yield();
    }
    for (int i = 0; i < 10000; ++i) {
      child.RecordKernelUsage("kernel");
    }
  });
  std::thread reader([&] {
    start.store(true);
    for (int i = 0; i < 1000; ++i) {
      const auto snapshot = rt.GetKernelUsage();
      EXPECT_LE(snapshot.size(), RuntimeContext::kKernelUsageLimit);
      for (const auto &name : snapshot) {
        EXPECT_EQ(name, "kernel");
      }
      rt.ClearKernelUsage();
    }
  });
  for (int i = 0; i < 1000; ++i) {
    rt.set_kernel_usage_enabled(false);
    const auto size = rt.GetKernelUsage().size();
    EXPECT_LE(child.GetKernelUsage().size(), size);
    rt.set_kernel_usage_enabled(true);
  }
  writer.join();
  reader.join();
  rt.set_kernel_usage_enabled(false);
  rt.ClearKernelUsage();
  child.RecordKernelUsage("ignored");
  EXPECT_TRUE(rt.GetKernelUsage().empty());
}

namespace {

class UsageRecordingKernel : public KernelBase {
public:
  explicit UsageRecordingKernel(const KernelContext &ctx) : KernelBase(ctx) {}

  void Run(RuntimeContext &rt) override {
    rt.RecordKernelUsage(node_->name());
    rt.Set(node_->output(0), rt.Get(node_->input(0)));
  }
};

} // namespace

TEST(KernelUsage, NestedFunctionAndSubgraphDispatchBelongsToSession) {
  const std::string domain = "test.kernel_usage";
  core::runtime::RegisterKernelFn(
      domain, "Record", core::symbolic::Device::kCPU,
      [](const NodeProto &node, RuntimeContext &rt) -> std::unique_ptr<KernelBase> {
        auto kernel = std::make_unique<UsageRecordingKernel>(rt.kernel_ctx());
        kernel->set_node(node);
        return kernel;
      });

  ModelProto model;
  FunctionProto *function = model.add_functions();
  function->set_domain(domain);
  function->set_name("Nested");
  function->add_input("condition");
  function->add_input("value");
  function->add_output("result");
  auto *record = function->add_node();
  record->set_domain(domain);
  record->set_op_type("Record");
  record->set_name("function");
  record->add_input("value");
  record->add_output("intermediate");
  auto *branch = function->add_node();
  branch->set_op_type("If");
  branch->add_input("condition");
  branch->add_output("result");
  for (const std::string name : {"then_branch", "else_branch"}) {
    auto *attribute = branch->add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto::AttributeType::GRAPH);
    auto *body = attribute->add_g();
    body->add_output()->set_name("branch_result");
    auto *node = body->add_node();
    node->set_domain(domain);
    node->set_op_type("Record");
    node->set_name(name);
    node->add_input("intermediate");
    node->add_output("branch_result");
  }
  auto *graph = model.add_graph();
  graph->add_input()->set_name("condition");
  graph->add_input()->set_name("x");
  graph->add_output()->set_name("y");
  auto *call = graph->add_node();
  call->set_domain(domain);
  call->set_op_type("Nested");
  call->add_input("condition");
  call->add_input("x");
  call->add_output("y");

  RuntimeContext first(KernelContext(core::runtime::DefaultOpset(18)));
  RuntimeContext second(KernelContext(core::runtime::DefaultOpset(18)));
  core::runtime::RegisterModelFunctions(model, first);
  core::runtime::RegisterModelFunctions(model, second);
  first.set_kernel_usage_enabled(true);
  second.set_kernel_usage_enabled(true);
  first.Set("condition", Tensor::FromBool("condition", {}, {true}));
  second.Set("condition", Tensor::FromBool("condition", {}, {false}));
  first.Set("x", Tensor::FromFloat("x", {1}, {1.0f}));
  second.Set("x", Tensor::FromFloat("x", {1}, {2.0f}));
  RuntimeSession first_session(first.GetExecutionPlan(*graph));
  RuntimeSession second_session(second.GetExecutionPlan(*graph));
  first_session.Run(first);
  second_session.Run(second);
  EXPECT_EQ(first.GetKernelUsage(), (std::vector<std::string>{"function", "then_branch"}));
  EXPECT_EQ(second.GetKernelUsage(), (std::vector<std::string>{"function", "else_branch"}));
  EXPECT_FLOAT_EQ(first.Get("y").AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(second.Get("y").AsFloat()[0], 2.0f);
  EXPECT_TRUE(first.custom_kernels().empty());
  EXPECT_TRUE(second.custom_kernels().empty());
}
