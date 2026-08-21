// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_proto/onnx.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace rt = ONNX_LIGHT_NAMESPACE::core::runtime;
using ONNX_LIGHT_NAMESPACE::GraphProto;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::ValueInfoProto;

GraphProto MakeAbsGraph() {
  GraphProto graph;
  graph.set_name("profile_abs");
  ValueInfoProto input;
  input.set_name("x");
  ValueInfoProto output;
  output.set_name("y");
  graph.ref_input().push_back(input);
  graph.ref_output().push_back(output);
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");
  graph.ref_node().push_back(node);
  return graph;
}

int main(int argc, char **argv) {
  const bool hardware_counters = argc == 2 && std::string_view(argv[1]) == "--hardware-counters";
  if (argc > 2 || (argc == 2 && !hardware_counters)) {
    std::cerr << "Usage: " << argv[0] << " [--hardware-counters]\n";
    return 1;
  }
  ONNX_LIGHT_NAMESPACE::onnx_kernels::RegisterKernelFunctions();
  const GraphProto graph = MakeAbsGraph();
  rt::RuntimeContext context(rt::KernelContext(rt::DefaultOpset(18)));
  context.Set("x", rt::Tensor::FromFloat("x", {100000}, std::vector<float>(100000, -1.0f)));

  auto collector = std::make_shared<rt::ParallelRegionCollector>(1, hardware_counters);
  rt::RuntimeSession session(context.GetExecutionPlan(graph),
                             rt::RuntimeSessionOptions{
                                 .parameters = rt::RuntimeParameters(hardware_counters ? 1 : 2),
                                 .parallel_region_collector = collector,
                             });
  session.Run(context);
  session.Run(context);

  const rt::ParallelRegionReport report = session.parallel_region_report();
  for (const rt::ParallelRegionReportEvent &event : report.events()) {
    std::cout << event.file_name << ":" << event.line << " requested=" << event.requested_threads
              << " admitted=" << event.admitted_threads << " observed=" << event.observed_threads
              << " wall_time_ns=" << event.wall_time_ns.value_or(0)
              << " counters=" << rt::HardwareCounterStatusName(event.counter_status);
    if (event.ipc.has_value()) {
      std::cout << " ipc=" << *event.ipc;
    }
    if (event.llc_miss_rate.has_value()) {
      std::cout << " llc_miss_rate=" << *event.llc_miss_rate;
    }
    if (!event.ipc.has_value() && !event.llc_miss_rate.has_value()) {
      std::cout << " (using portable timing)";
    } else {
      std::cout << " (hardware metrics available)";
    }
    std::cout << "\n";
  }
  std::cout << "dropped_events=" << report.dropped_events() << "\n";
  return 0;
}
