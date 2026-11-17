// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Builds with ONNX_LIGHT_BUILD_BENCHMARKS=ON and runs bench_local_function.
// Times warmed RuntimeSession::Run with one scalar Neg/Abs pair, either inline
// or in a local function. Only unrelated live caller tensors vary; their setup
// and output collection are not timed. These are synthetic session timings,
// not an isolated measurement of binding overhead or a real-model result.

#include "onnx_core/runtime/runtime_session.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

ModelProto MakeModel(bool extracted) {
  ModelProto model;
  model.set_ir_version(10);
  auto *opset = model.add_opset_import();
  opset->set_version(18);
  auto *local_opset = model.add_opset_import();
  local_opset->set_domain("local");
  local_opset->set_version(1);
  auto *function = model.add_functions();
  function->set_name("NegAbs");
  function->set_domain("local");
  *function->add_opset_import() = model.opset_import(0);
  function->add_input("x");
  function->add_output("y");
  auto *neg = function->add_node();
  neg->set_op_type("Neg");
  neg->add_input("x");
  neg->add_output("tmp");
  auto *abs = function->add_node();
  abs->set_op_type("Abs");
  abs->add_input("tmp");
  abs->add_output("y");
  auto *graph = model.add_graph();
  graph->set_name("local_function_scaling");
  graph->add_input()->set_name("x");
  graph->add_output()->set_name("y");
  if (extracted) {
    auto *call = graph->add_node();
    call->set_op_type("NegAbs");
    call->set_domain("local");
    call->add_input("x");
    call->add_output("y");
  } else {
    for (const auto &node : function->node())
      *graph->add_node() = node;
  }
  return model;
}

double Measure(bool extracted, size_t unrelated_values) {
  ModelProto model = MakeModel(extracted);
  CpuExecutionPolicy execution;
  execution.num_threads = 1;
  RuntimeSession session(model, RuntimeSessionOptions{.cpu_execution = execution});
  RuntimeContext rt;
  rt.set_release_intermediates(true);
  rt.Put("x", Tensor::FromFloat("x", {}, {-2}));
  for (size_t i = 0; i < unrelated_values; ++i) {
    const std::string name = "unrelated_" + std::to_string(i);
    rt.Put(name, Tensor::FromFloat(name, {}, {3}));
  }
  for (size_t i = 0; i < 20; ++i)
    session.Run(rt);
  constexpr size_t iterations = 2000;
  std::vector<double> samples;
  for (size_t sample = 0; sample < 7; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i)
      session.Run(rt);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    samples.push_back(std::chrono::duration<double, std::nano>(elapsed).count() / iterations);
  }
  if (rt.Get("y").AsFloat()[0] != 2 || std::as_const(rt).tensors().size() != unrelated_values + 2)
    throw std::runtime_error("incorrect benchmark result or caller values lost");
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

} // namespace

int main() {
  onnx_kernels::RegisterKernelFunctions();
  std::cout << "unrelated_live_values,inline_median_ns,local_function_median_ns\n";
  std::cout << std::fixed << std::setprecision(1);
  for (const size_t count : {0, 32, 512, 4096})
    std::cout << count << ',' << Measure(false, count) << ',' << Measure(true, count) << '\n';
}
