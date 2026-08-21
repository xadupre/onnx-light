/**
 * Compares disabled and enabled ParallelFor profiling overhead.
 *
 * Build:
 *   cmake -B build -DONNX_LIGHT_BUILD_BENCHMARKS=ON
 *   cmake --build build --target bench_parallel_for_profiling -j
 *
 * Usage:
 *   ./build/bench_parallel_for_profiling [-n calls-per-sample]
 */

#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_core/runtime/tuning/parallel_region_collector.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

constexpr int64_t kIterations = 1 << 20;
constexpr size_t kSamples = 21;

double MedianNanoseconds(uint64_t calls, ParallelRegionCollector *collector, uint64_t &checksum) {
  std::vector<double> samples;
  samples.reserve(kSamples);
  for (size_t sample = 0; sample < kSamples; ++sample) {
    ParallelRegionCollectorScope collector_scope(collector);
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t call = 0; call < calls; ++call) {
      ParallelFor(
          kIterations,
          [&checksum](int64_t begin, int64_t end) {
            checksum += static_cast<uint64_t>(end - begin);
          },
          "large-loop");
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    samples.push_back(std::chrono::duration<double, std::nano>(elapsed).count() /
                      static_cast<double>(calls));
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  uint64_t calls = 10000;
  if (argc == 3 && std::string(argv[1]) == "-n") {
    calls = std::strtoull(argv[2], nullptr, 10);
  } else if (argc != 1) {
    std::cerr << "Usage: " << argv[0] << " [-n calls-per-sample]\n";
    return 1;
  }
  if (calls == 0) {
    std::cerr << "calls-per-sample must be positive\n";
    return 1;
  }

  CpuExecutionPolicy policy;
  policy.num_threads = 1;
  policy.affinity_policy = CpuAffinityPolicy::kNone;
  CpuExecutorRegistry registry(1);
  const std::shared_ptr<CpuExecutor> executor = registry.Acquire(policy);
  CpuExecutorScope executor_scope(executor.get());
  ParallelRegionCollector collector(1);
  uint64_t checksum = 0;

  const double disabled = MedianNanoseconds(calls, nullptr, checksum);
  const double enabled = MedianNanoseconds(calls, &collector, checksum);
  std::cout << "calls per sample: " << calls << "\n"
            << "disabled median: " << disabled << " ns/call\n"
            << "enabled median: " << enabled << " ns/call\n"
            << "enabled / disabled: " << enabled / disabled << "\n"
            << "dropped events: " << collector.dropped_events() << "\n"
            << "checksum: " << checksum << "\n";
  return 0;
}
