// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Runs on x86-64 and ARM64; compare policies before changing portable thresholds.
#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_extensions/kernels/kernels/nn/variance_accumulator.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

int main(int argc, char **argv) {
  int64_t repetitions = 100;
  if (argc == 3 && std::string(argv[1]) == "-n") {
    repetitions = std::strtoll(argv[2], nullptr, 10);
  } else if (argc != 1) {
    std::cerr << "Usage: " << argv[0] << " [-n repetitions]\n";
    return 1;
  }
  if (repetitions <= 0) {
    std::cerr << "repetitions must be positive\n";
    return 1;
  }

  core::runtime::CpuExecutionPolicy policy;
  policy.num_threads = 2;
  policy.affinity_policy = core::runtime::CpuAffinityPolicy::kNone;
  const auto executor = core::runtime::GlobalCpuExecutorRegistry().Acquire(policy);
  const core::runtime::CpuExecutorScope scope(executor.get());
  double checksum = 0;
  for (int64_t size : {int64_t{128}, int64_t{32768}, int64_t{262144}}) {
    std::vector<float> values(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
      values[static_cast<size_t>(i)] = 1e8f + static_cast<float>(i % 17) * 8.0f;
    }
    for (int64_t minimum : {core::runtime::kParallelForGrainSize, int64_t{1}}) {
      const auto start = std::chrono::steady_clock::now();
      for (int64_t repetition = 0; repetition < repetitions; ++repetition) {
        const auto stats = onnx_kernels::kernel::statistics::Accumulate<double>(
            size, [&](int64_t i) { return values[static_cast<size_t>(i)]; }, minimum);
        checksum += stats.Variance();
      }
      const auto duration = std::chrono::steady_clock::now() - start;
      std::cout << "size=" << size << " minimum=" << minimum << " ns/reduction="
                << std::chrono::duration<double, std::nano>(duration).count() / repetitions << '\n';
    }
  }
  std::cout << "checksum=" << checksum << '\n';
}
