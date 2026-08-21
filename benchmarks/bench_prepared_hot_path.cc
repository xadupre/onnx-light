// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/compute/prepared_execution.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

uint64_t Replay(const ExecutionPlan &plan) {
  uint64_t value = 0;
  for (const ExecuteAction &action : plan.actions()) {
    value += static_cast<uint64_t>(action.kind()) + action.node_index() + 1;
  }
  return value;
}

} // namespace

int main(int argc, char **argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 100000;
  const double maximum_overhead_ratio = argc > 2 ? std::atof(argv[2]) : 50.0;
  if (iterations <= 0 || maximum_overhead_ratio <= 0) {
    std::cerr << "iterations and maximum overhead ratio must be positive\n";
    return 1;
  }
  GraphProto graph;
  graph.add_input()->set_name("X");
  graph.add_output()->set_name("Y");
  std::string input = "X";
  for (size_t i = 0; i < 8; ++i) {
    NodeProto *node = graph.add_node();
    node->set_op_type("Identity");
    node->add_input(input);
    input = i == 7 ? "Y" : "v" + std::to_string(i);
    node->add_output(input);
  }
  const ExecutionPlan direct_plan(graph);
  TaskDescriptor replay{
      TaskId{2}, TaskScope::kInvocation, TaskKind::kExecute, ResourceClass::kCpu, {TaskId{1}}};
  replay.actions = ActionRange{0, direct_plan.actions().size()};
  const PreparedExecutionPlan prepared_plan(
      {TaskDescriptor{TaskId{1}, TaskScope::kSession, TaskKind::kPrepare, ResourceClass::kInline},
       replay});
  PreparedExecutionState state;
  uint64_t prepared_checksum = 0;
  const auto prepared_executor = [&](const TaskDescriptor &task, PreparedExecutionState &) {
    if (task.scope == TaskScope::kInvocation) {
      prepared_checksum += Replay(direct_plan);
    }
  };
  prepared_plan.RunSequential(state, prepared_executor);

  uint64_t direct_checksum = 0;
  const auto direct_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    direct_checksum += Replay(direct_plan);
  }
  const auto direct_end = std::chrono::steady_clock::now();
  const auto prepared_start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    prepared_plan.RunSequential(state, prepared_executor);
  }
  const auto prepared_end = std::chrono::steady_clock::now();

  const double direct_ns =
      std::chrono::duration<double, std::nano>(direct_end - direct_start).count() / iterations;
  const double prepared_ns =
      std::chrono::duration<double, std::nano>(prepared_end - prepared_start).count() / iterations;
  const double overhead_ratio = prepared_ns / direct_ns;
  std::cout << "direct_execution_plan_ns=" << direct_ns << '\n'
            << "prepared_hot_path_ns=" << prepared_ns << '\n'
            << "overhead_ratio=" << overhead_ratio << '\n'
            << "maximum_overhead_ratio=" << maximum_overhead_ratio << '\n'
            << "checksum=" << direct_checksum + prepared_checksum << '\n';
  return overhead_ratio <= maximum_overhead_ratio ? 0 : 2;
}
