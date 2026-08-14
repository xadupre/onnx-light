/**
 * Benchmarks cold kernel tuning resolution separately from steady-state execution.
 *
 * Build:
 *   cmake -B build -DONNX_LIGHT_BUILD_BENCHMARKS=ON
 *   cmake --build build --target bench_kernel_tuning -j
 *
 * Usage:
 *   ./build/bench_kernel_tuning [-n iterations]
 */

#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernel_tuning.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_proto/onnx_helper.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

constexpr const char *kDomain = "benchmark.onnx_light.kernel_tuning";

KernelTuningKey MakeKey() {
  return {"onnx_light_benchmark", "TunedIncrement", "portable", DataType::INT64, Device::kCPU, 1};
}

KernelTuningParameters MakeParameters() {
  return {MakeKey(), {{"algorithm.increment", int64_t{3}}}};
}

CpuExecutionDescriptor MakeExecution() { return {core::platform::GetCpuDescriptor(), 1}; }

class TunedIncrementKernel final : public KernelBase {
public:
  TunedIncrementKernel(const NodeProto &node, const KernelContext &context, KernelTuningKey key)
      : KernelBase(context), key_(std::move(key)) {
    set_node(node);
  }

  KernelTuningKey TuningKey(int32_t element_type) const override {
    KernelTuningKey key = key_;
    key.element_type = element_type;
    return key;
  }

  void Configure(const KernelTuningParameters &parameters) override {
    increment_ = parameters.Get<int64_t>("algorithm.increment");
  }

  void Run(RuntimeContext &runtime) override {
    const int64_t value = runtime.Get(node_->input(0)).AsInt64()[0];
    runtime.tensors()[node_->output(0)] =
        Tensor::FromInt64(node_->output(0), {1}, {value + increment_});
  }

private:
  KernelTuningKey key_;
  int64_t increment_ = 1;
};

double NanosecondsPerIteration(std::chrono::steady_clock::duration duration, uint64_t iterations) {
  return std::chrono::duration<double, std::nano>(duration).count() /
         static_cast<double>(iterations);
}

} // namespace

int main(int argc, char **argv) {
  uint64_t iterations = 100000;
  if (argc == 3 && std::string(argv[1]) == "-n") {
    iterations = std::strtoull(argv[2], nullptr, 10);
  } else if (argc != 1) {
    std::cerr << "Usage: " << argv[0] << " [-n iterations]\n";
    return 1;
  }
  if (iterations == 0) {
    std::cerr << "iterations must be positive\n";
    return 1;
  }

  KernelTuningRegistry registry;
  const KernelTuningKey key = MakeKey();
  registry.RegisterSchema(KernelTuningSchema(MakeParameters()));
  core::platform::CpuSelector selector;
  selector.minimum_threads = 1;
  KernelTuningParameters selected = MakeParameters();
  selected.values["algorithm.increment"] = int64_t{5};
  registry.RegisterProfile(key, selector, selected);
  const CpuExecutionDescriptor execution = MakeExecution();

  uint64_t checksum = 0;
  const KernelTuningRegistryAccessCounts cold_before = registry.AccessCounts();
  const auto snapshot_start = std::chrono::steady_clock::now();
  for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
    const KernelTuningRegistrySnapshot snapshot = registry.Snapshot();
    checksum += snapshot.generation();
  }
  const auto snapshot_duration = std::chrono::steady_clock::now() - snapshot_start;

  const KernelTuningRegistrySnapshot snapshot = registry.Snapshot();
  const auto resolution_start = std::chrono::steady_clock::now();
  for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
    const KernelTuningParameters *parameters = snapshot.Resolve(key, execution);
    if (parameters == nullptr) {
      std::cerr << "cold resolution did not find the registered profile\n";
      return 2;
    }
    checksum += 1;
  }
  const auto resolution_duration = std::chrono::steady_clock::now() - resolution_start;
  const KernelTuningRegistryAccessCounts cold_after = registry.AccessCounts();
  if (cold_after.snapshots - cold_before.snapshots != iterations + 1 ||
      cold_after.lookups - cold_before.lookups != iterations ||
      cold_after.resolutions - cold_before.resolutions != iterations) {
    std::cerr << "cold resolution access counters are inconsistent\n";
    return 2;
  }

  RegisterKernelTuningSchema(KernelTuningSchema(MakeParameters()));
  RegisterKernelFn(
      kDomain, "TunedIncrement", Device::kCPU,
      [](const NodeProto &node, RuntimeContext &runtime) -> std::unique_ptr<KernelBase> {
        return std::make_unique<TunedIncrementKernel>(node, runtime.kernel_ctx(), MakeKey());
      });

  GraphProto graph;
  graph.add_input()->set_name("x");
  graph.add_output()->set_name("y");
  graph.add_node(MakeNode("TunedIncrement", {"x"}, {"y"}, kDomain));
  RuntimeContext runtime(KernelContext(DefaultOpset(18)));
  runtime.Set("x", Tensor::FromInt64("x", {1}, {7}));
  RuntimeSession session(runtime.GetExecutionPlan(graph));
  session.Run(runtime);

  const KernelTuningResolutionStatistics cold_session = session.tuning_resolution_statistics();
  const KernelTuningRegistryAccessCounts hot_before = GetKernelTuningRegistry().AccessCounts();
  const auto hot_start = std::chrono::steady_clock::now();
  for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
    session.Run(runtime);
  }
  const auto hot_duration = std::chrono::steady_clock::now() - hot_start;
  const KernelTuningRegistryAccessCounts hot_after = GetKernelTuningRegistry().AccessCounts();
  if (hot_after != hot_before) {
    std::cerr << "steady-state execution accessed the kernel tuning registry\n";
    return 2;
  }
  checksum += static_cast<uint64_t>(runtime.Get("y").AsInt64()[0]);

  std::cout << "iterations: " << iterations << "\n"
            << "cold registry snapshot: " << NanosecondsPerIteration(snapshot_duration, iterations)
            << " ns/iteration\n"
            << "cold profile resolution: "
            << NanosecondsPerIteration(resolution_duration, iterations) << " ns/iteration\n"
            << "first session tuning: " << cold_session.TotalDurationNs() << " ns ("
            << cold_session.tunable_kernels << " kernel)\n"
            << "steady RuntimeSession::Run: " << NanosecondsPerIteration(hot_duration, iterations)
            << " ns/iteration\n"
            << "steady registry accesses: snapshots=0, lookups=0, resolutions=0\n"
            << "checksum: " << checksum << "\n";
  return 0;
}
