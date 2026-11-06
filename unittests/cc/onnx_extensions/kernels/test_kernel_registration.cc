// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_session.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::runtime;
using core::shapes::PeakMemoryEstimate;
using core::symbolic::Device;
using core::symbolic::SymDim;
using core::symbolic::SymShape;

namespace {

KernelRegistration Implementation(int64_t bytes, float value = 1) {
  return {MakeCustomKernelFactory([value](const NodeProto &node, RuntimeContext &rt) {
            rt.Put(node.output(0), Tensor::FromFloat(node.output(0), {1}, {value}, rt.allocator()));
          }),
          [bytes](Device, const NodeProto &, const std::vector<SymShape> &) {
            return PeakMemoryEstimate::Exact(SymDim(bytes));
          }};
}

int64_t GlobalMemory(const std::string &op, Device device = Device::kCPU,
                     const std::string &domain = "registration_test") {
  return core::shapes::ComputePeakMemory(domain, op, device, {}).worst_case.AsInt();
}

GraphProto Graph(const std::string &op) {
  GraphProto graph;
  graph.add_output()->set_name("Y");
  auto *node = graph.add_node();
  node->set_domain("registration_test");
  node->set_op_type(op);
  node->add_output("Y");
  return graph;
}

} // namespace

TEST(KernelRegistration, ReplacementRefusalAliasesAndDevices) {
  const std::string op = "Replacement";
  ASSERT_TRUE(RegisterKernelFn("", op, Device::kCPU, Implementation(11)));
  EXPECT_FALSE(RegisterKernelFn("ai.onnx", op, Device::kUndefined, Implementation(22), false));
  EXPECT_FALSE(RegisterKernelFn("ai.onnx", op, Device::kCPU, KernelRegistration{}, false));
  EXPECT_EQ(GlobalMemory(op, Device::kUndefined, ""), 11);
  ASSERT_TRUE(RegisterKernelFn("ai.onnx", op, Device::kUndefined, Implementation(33)));
  EXPECT_EQ(GlobalMemory(op, Device::kCPU, ""), 33);
  ASSERT_TRUE(RegisterKernelFn("", op, Device::kGPU0, Implementation(44)));
  EXPECT_EQ(GlobalMemory(op, Device::kGPU0, ""), 44);
  EXPECT_EQ(GlobalMemory(op, Device::kCPU, ""), 33);
}

TEST(KernelRegistration, MissingEstimatorExplicitlySelectsZero) {
  const std::string op = "NoEstimator";
  ASSERT_TRUE(RegisterKernelFn("registration_test", op, Device::kCPU, Implementation(11)));
  auto registration = Implementation(22);
  registration.peak_memory = {};
  EXPECT_FALSE(RegisterKernelFn("registration_test", op, Device::kCPU, registration, false));
  EXPECT_EQ(GlobalMemory(op), 11);
  ASSERT_TRUE(RegisterKernelFn("registration_test", op, Device::kCPU, registration));
  EXPECT_EQ(GlobalMemory(op), 0);
  core::shapes::RegisterComputePeakMemoryFn("registration_test", op, Device::kCPU,
                                            Implementation(33).peak_memory, false);
  EXPECT_EQ(GlobalMemory(op), 0);
  core::shapes::ShapesContext incomplete;
  auto graph = Graph(op);
  graph.mutable_node(0)->add_input("missing");
  EXPECT_EQ(
      core::shapes::ComputePeakMemory(incomplete, Device::kCPU, graph.node(0)).worst_case.AsInt(),
      0);
  ASSERT_TRUE(RegisterKernelFn("registration_test", op, Device::kCPU, Implementation(44)));
  ASSERT_TRUE(RegisterKernelFn("registration_test", op, Device::kCPU, registration.factory));
  EXPECT_EQ(GlobalMemory(op), 0);
}

TEST(KernelRegistration, CollectionAndFactoryConstructionArePure) {
  ASSERT_TRUE(RegisterKernelFn("registration_test", "Pure", Device::kCPU, Implementation(61, 7)));
  const auto kernels = KernelDispatchTable();
  const auto estimators = core::shapes::PeakMemoryDispatchTable();
  const auto &builtins = onnx_kernels::KernelFunctions();
  const auto &builtin_estimators = onnx_kernels::PeakMemoryFunctions();
  ASSERT_NE(builtins.find("ai.onnx:MatMul"), builtins.end());
  ASSERT_NE(builtin_estimators.find("ai.onnx:MatMul"), builtin_estimators.end());
  auto factory = MakeKernelFactory(Implementation(72));
  EXPECT_EQ(KernelDispatchTable().size(), kernels.size());
  EXPECT_EQ(core::shapes::PeakMemoryDispatchTable().size(), estimators.size());
  for (const auto &[key, fn] : kernels)
    EXPECT_EQ(KernelDispatchTable().at(key).target_type(), fn.target_type());
  for (const auto &[key, fn] : estimators)
    EXPECT_EQ(core::shapes::PeakMemoryDispatchTable().at(key).target_type(), fn.target_type());
  EXPECT_EQ(GlobalMemory("Pure"), 61);
  RuntimeContext rt;
  auto graph = Graph("Pure");
  auto kernel =
      KernelDispatchTable().at("registration_test:Pure")(graph.node(0), rt, KernelContext{});
  kernel->Run(rt);
  EXPECT_FLOAT_EQ(rt.Get("Y").AsFloat()[0], 7);
  ASSERT_TRUE(RegisterKernelFn("registration_test", "Collected", Device::kCPU, factory));
  EXPECT_EQ(GlobalMemory("Collected"), 72);
}

class KernelRegistrationOrder : public ::testing::TestWithParam<bool> {};

TEST_P(KernelRegistrationOrder, LazyBuiltinsPreserveExplicitImplementationsAndAccounting) {
  struct RestoreBuiltins {
    NodeKernelFn matmul = onnx_kernels::KernelFunctions().at("ai.onnx:MatMul");
    NodeKernelFn gather = onnx_kernels::KernelFunctions().at("ai.onnx:Gather");
    ~RestoreBuiltins() {
      RegisterKernelFn("", "MatMul", Device::kCPU, matmul);
      RegisterKernelFn("", "Gather", Device::kCPU, gather);
    }
  } restore;
  ASSERT_TRUE(RegisterKernelFn("", "MatMul", Device::kCPU, Implementation(123)));
  ASSERT_TRUE(RegisterKernelFn("", "Gather", Device::kCPU, Implementation(1).factory));
  if (GetParam())
    onnx_kernels::RegisterPeakMemoryFunctions();
  onnx_kernels::RegisterKernelFunctions();
  onnx_kernels::RegisterPeakMemoryFunctions();
  EXPECT_EQ(GlobalMemory("MatMul", Device::kCPU, ""), 123);
  EXPECT_EQ(GlobalMemory("Gather", Device::kCPU, ""), 0);
  NodeProto node;
  node.set_op_type("MatMul");
  node.add_output("Y");
  RuntimeContext rt;
  auto kernel = KernelDispatchTable().at("ai.onnx:MatMul")(node, rt, KernelContext{});
  kernel->Run(rt);
  EXPECT_FLOAT_EQ(rt.Get("Y").AsFloat()[0], 1);
  EXPECT_EQ(kernel->ComputePeakMemory(Device::kCPU, {}).worst_case.AsInt(), 123);
  ASSERT_TRUE(RegisterKernelFn("", "MatMul", Device::kCPU, restore.matmul));
  ASSERT_TRUE(RegisterKernelFn("", "Gather", Device::kCPU, restore.gather));
  EXPECT_TRUE(core::shapes::ComputePeakMemory("", "MatMul", Device::kCPU,
                                              {SymShape({2, 3}), SymShape({3, 4})})
                  .worst_case.AsInt() > 0);
}

INSTANTIATE_TEST_SUITE_P(RegistrationOrders, KernelRegistrationOrder, ::testing::Bool());

TEST(KernelRegistration, SessionsRetainResolvedFactoryAndEstimator) {
  const std::string op = "Session";
  auto graph = Graph(op);
  ExecutionPlan plan(graph);
  RuntimeContext global_rt;
  RuntimeContext local_rt;
  RuntimeSession old_session(plan);
  RuntimeSession local_session(plan);
  EXPECT_THROW(old_session.ComputePeakMemory(0, {}), std::invalid_argument);
  ASSERT_TRUE(RegisterKernelFn("registration_test", op, Device::kCPU, Implementation(10, 1)));
  old_session.Run(global_rt);
  ASSERT_TRUE(
      local_rt.RegisterKernelFn("registration_test", op, Device::kCPU, Implementation(20, 2)));
  EXPECT_FALSE(local_rt.RegisterKernelFn("registration_test", op, Device::kCPU,
                                         Implementation(99, 9), false));
  EXPECT_FALSE(local_rt.RegisterKernelFn("registration_test", op, Device::kCPU,
                                         KernelRegistration{}, false));
  local_session.Run(local_rt);
  EXPECT_EQ(GlobalMemory(op), 10);
  EXPECT_EQ(old_session.ComputePeakMemory(0, {}).worst_case.AsInt(), 10);
  EXPECT_EQ(local_session.ComputePeakMemory(0, {}).worst_case.AsInt(), 20);
  ASSERT_TRUE(RegisterKernelFn("registration_test", op, Device::kCPU, Implementation(30, 3)));
  ASSERT_TRUE(
      local_rt.RegisterKernelFn("registration_test", op, Device::kCPU, Implementation(40, 4)));
  old_session.Run(global_rt);
  local_session.Run(local_rt);
  EXPECT_FLOAT_EQ(global_rt.Get("Y").AsFloat()[0], 1);
  EXPECT_FLOAT_EQ(local_rt.Get("Y").AsFloat()[0], 2);
  EXPECT_EQ(old_session.ComputePeakMemory(0, {}).worst_case.AsInt(), 10);
  EXPECT_EQ(local_session.ComputePeakMemory(0, {}).worst_case.AsInt(), 20);
  RuntimeSession new_session(plan);
  RuntimeSession new_local_session(plan);
  new_session.Run(global_rt);
  new_local_session.Run(local_rt);
  EXPECT_FLOAT_EQ(global_rt.Get("Y").AsFloat()[0], 3);
  EXPECT_FLOAT_EQ(local_rt.Get("Y").AsFloat()[0], 4);
  EXPECT_EQ(new_session.ComputePeakMemory(0, {}).worst_case.AsInt(), 30);
  EXPECT_EQ(new_local_session.ComputePeakMemory(0, {}).worst_case.AsInt(), 40);
  EXPECT_EQ(GlobalMemory(op), 30);
  ASSERT_TRUE(local_rt.RegisterKernelFn("registration_test", op, Device::kCPU,
                                        Implementation(50, 5).factory));
  RuntimeSession zero_session(plan);
  zero_session.Run(local_rt);
  EXPECT_FLOAT_EQ(local_rt.Get("Y").AsFloat()[0], 5);
  EXPECT_EQ(zero_session.ComputePeakMemory(0, {}).worst_case.AsInt(), 0);
  EXPECT_EQ(GlobalMemory(op), 30);
  EXPECT_THROW(old_session.ComputePeakMemory(std::numeric_limits<size_t>::max(), {}),
               std::invalid_argument);
  local_rt.ClearCustomKernels();
  local_session.Run(local_rt);
  EXPECT_FLOAT_EQ(local_rt.Get("Y").AsFloat()[0], 2);
  EXPECT_EQ(local_session.ComputePeakMemory(0, {}).worst_case.AsInt(), 20);
}

TEST(KernelRegistration, ValidationAppliesToResolvedEstimatesAndFactories) {
  auto registration = Implementation(-1);
  auto graph = Graph("Invalid");
  RuntimeContext rt;
  auto kernel = MakeKernelFactory(registration)(graph.node(0), rt, KernelContext{});
  EXPECT_THROW(kernel->ComputePeakMemory(Device::kCPU, {}), std::invalid_argument);
  registration.peak_memory = [](Device, const NodeProto &, const std::vector<SymShape> &) {
    return PeakMemoryEstimate{SymDim(int64_t{1}), SymDim(int64_t{2})};
  };
  kernel = MakeKernelFactory(registration)(graph.node(0), rt, KernelContext{});
  EXPECT_THROW(kernel->ComputePeakMemory(Device::kCPU, {}), std::invalid_argument);
  registration.peak_memory = [](Device, const NodeProto &node, const std::vector<SymShape> &) {
    return core::shapes::UnknownPeakMemory(node.domain(), node.op_type());
  };
  kernel = MakeKernelFactory(registration)(graph.node(0), rt, KernelContext{});
  EXPECT_TRUE(kernel->ComputePeakMemory(Device::kCPU, {}).worst_case.IsExpr());
  EXPECT_THROW(MakeKernelFactory(KernelRegistration{}), std::invalid_argument);
  registration.factory = [](const NodeProto &, RuntimeContext &, const KernelContext &) {
    return std::unique_ptr<KernelBase>{};
  };
  auto factory = MakeKernelFactory(registration);
  EXPECT_THROW(factory(graph.node(0), rt, KernelContext{}), std::invalid_argument);
}

TEST(KernelRegistration, AttentionSessionsRejectIncompleteNamedOptionalShapes) {
  for (const auto *op : {"Attention", "LinearAttention"}) {
    for (int slot = 3; slot < 6; ++slot) {
      SCOPED_TRACE(testing::Message() << op << "," << slot);
      ModelProto model;
      model.add_opset("", std::string(op) == "Attention" ? 23 : 27);
      auto &graph = *model.mutable_graph();
      auto &node = *graph.add_node();
      node.set_op_type(op);
      node.add_output("Y");
      graph.add_output()->set_name("Y");
      for (const auto *name : {"q", "k", "v"}) {
        node.add_input(name);
        graph.add_input()->set_name(name);
      }
      while (node.input_size() <= slot)
        node.add_input("");
      *node.mutable_input(slot) = "optional";
      graph.add_input()->set_name("optional");
      RuntimeContext rt;
      ASSERT_TRUE(rt.RegisterKernelFn(
          "", op, Device::kCPU, onnx_kernels::KernelFunctions().at(std::string("ai.onnx:") + op)));
      RuntimeSession session(model);
      session.DeclarePreparations(rt);
      const SymShape shape =
          std::string(op) == "Attention" ? SymShape({1, 1, 1, 1}) : SymShape({1, 1, 1});
      std::vector<SymShape> inputs(3, shape);
      const auto unknown = core::shapes::UnknownPeakMemory("", op);
      EXPECT_EQ(session.ComputePeakMemory(0, inputs), unknown);
      inputs.resize(slot + 1);
      EXPECT_EQ(session.ComputePeakMemory(0, inputs), unknown);
    }
  }
}
