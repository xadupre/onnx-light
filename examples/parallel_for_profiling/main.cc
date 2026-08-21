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

int main() {
  ONNX_LIGHT_NAMESPACE::onnx_kernels::RegisterKernelFunctions();
  const GraphProto graph = MakeAbsGraph();
  rt::RuntimeContext context(rt::KernelContext(rt::DefaultOpset(18)));
  context.Set("x", rt::Tensor::FromFloat("x", {100000}, std::vector<float>(100000, -1.0f)));

  auto collector = std::make_shared<rt::ParallelRegionCollector>(1);
  rt::RuntimeSession session(context.GetExecutionPlan(graph),
                             rt::RuntimeSessionOptions{
                                 .parameters = rt::RuntimeParameters(2),
                                 .parallel_region_collector = collector,
                             });
  session.Run(context);
  session.Run(context);

  const rt::ParallelRegionReport report = session.parallel_region_report();
  for (const rt::ParallelRegionReportEvent &event : report.events()) {
    std::cout << event.file_name << ":" << event.line << " requested=" << event.requested_threads
              << " admitted=" << event.admitted_threads << " observed=" << event.observed_threads
              << " wall_time_ns=" << event.wall_time_ns.value_or(0) << "\n";
  }
  std::cout << "dropped_events=" << report.dropped_events() << "\n";
  return 0;
}
