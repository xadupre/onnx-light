// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/tuning/kernel_tuning_cache.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

constexpr std::array<int32_t, 4> kTypes = {DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT16,
                                           DataType::BFLOAT16};

struct Workload {
  const char *name;
  int64_t m, n, k, trans_a, trans_b;
};

constexpr std::array<Workload, 10> kWorkloads = {{
    {"small", 16, 32, 32, 0, 0},
    {"skinny", 4, 256, 128, 0, 0},
    {"square", 128, 128, 128, 0, 0},
    {"large", 256, 256, 256, 0, 0},
    {"deep", 32, 64, 513, 0, 0},
    {"wide", 32, 512, 128, 0, 0},
    {"tall", 256, 64, 128, 0, 0},
    {"tails", 65, 129, 67, 0, 0},
    {"transpose_a", 65, 129, 67, 1, 0},
    {"transpose_b", 65, 129, 67, 0, 1},
}};

Tensor Input(int32_t dtype, const Shape &shape, int seed) {
  const size_t count = static_cast<size_t>(shape[0] * shape[1]);
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = static_cast<float>(static_cast<int>((i * 13 + seed) % 17) - 8) / 16;
  }
  if (dtype == DataType::DOUBLE) {
    return Tensor::FromDouble("", shape, std::vector<double>(values.begin(), values.end()));
  }
  Tensor result = Tensor::FromFloat("", shape, values);
  return dtype == DataType::FLOAT ? std::move(result) : DemoteFromFloat32(result, dtype);
}

void PrintDescriptor(const CpuExecutionDescriptor &execution,
                     const ResolvedCpuExecutionPolicy &policy, std::string_view revision) {
  const auto &cpu = execution.processor;
  std::cout << "# revision=" << revision << "\n# architecture=" << cpu.architecture
            << " vendor=" << cpu.vendor << " family=" << cpu.family.value_or(0)
            << " model=" << cpu.model.value_or(0) << " stepping=" << cpu.stepping.value_or(0)
            << " microarchitecture=" << std::quoted(cpu.microarchitecture)
            << " features=" << cpu.features.bits()
            << "\n# cache_line_bytes=" << cpu.cache_line_bytes.value_or(0)
            << " l1_data_bytes=" << cpu.l1_data_bytes.value_or(0)
            << " l2_bytes=" << cpu.l2_bytes.value_or(0) << " l3_bytes=" << cpu.l3_bytes.value_or(0)
            << " physical_cores=" << cpu.physical_cores.value_or(0)
            << " logical_cores=" << cpu.logical_cores.value_or(0)
            << "\n# requested_threads=" << policy.request.num_threads
            << " effective_threads=" << execution.effective_threads
            << " affinity=none spin=adaptive spin_iterations=" << policy.spin.iterations
            << " spin_duration_ns=" << policy.spin.duration_ns << " uses_smt=" << policy.uses_smt
            << " uses_efficiency_cores=" << policy.uses_efficiency_cores
            << " nested=" << policy.allow_nested_parallelism << "\n# visible_processors=";
  const char *separator = "";
  for (const auto &processor : ProcessVisibleLogicalProcessors()) {
    std::cout << separator << processor.group << ':' << processor.id;
    separator = " ";
  }
  std::cout << '\n';
}

void Measure() {
  using onnx_kernels::kernel::Gemm;
  using namespace onnx_kernels::tuning;
  std::cout << "dtype,workload,m,n,k,trans_a,trans_b,configuration,tile_m,tile_n,tile_k,"
               "pack_b_minimum_elements,repetition,duration_ns\n";
  for (int32_t dtype : kTypes) {
    const KernelContext context{DefaultOpset(13)};
    Gemm reference{context};
    const auto key = reference.TuningKey(dtype);
    const auto defaults = GetKernelTuningRegistry().FindSchema(key)->portable_defaults();
    for (const auto &workload : kWorkloads) {
      const Tensor a = Input(
          dtype, workload.trans_a ? Shape{workload.k, workload.m} : Shape{workload.m, workload.k},
          5);
      const Tensor b = Input(
          dtype, workload.trans_b ? Shape{workload.n, workload.k} : Shape{workload.k, workload.n},
          6);
      const Tensor expected = reference(a, b, nullptr, 1, 0, workload.trans_a, workload.trans_b);
      Tensor output = MakeOutputTensor(dtype, expected.shape, expected.size_bytes(), nullptr);
      std::vector<Gemm> candidates;
      candidates.reserve(kGemmAlgorithmConfigurationCount);
      for (int64_t configuration = 0; configuration < kGemmAlgorithmConfigurationCount;
           ++configuration) {
        candidates.emplace_back(context);
        auto parameters = defaults;
        parameters.values[kGemmAlgorithmConfiguration] = configuration;
        candidates.back().Configure(parameters);
        candidates.back()(a, b, nullptr, 1, 0, workload.trans_a, workload.trans_b, output);
        if (std::memcmp(expected.bytes(), output.bytes(), output.size_bytes()) != 0) {
          throw std::runtime_error("Gemm candidate output differs from portable baseline.");
        }
      }
      // Rotates the first timed configuration to reduce fixed-order bias.
      for (int repetition = 0; repetition < 7; ++repetition) {
        for (int64_t offset = 0; offset < kGemmAlgorithmConfigurationCount; ++offset) {
          const int64_t configuration = (offset + repetition) % kGemmAlgorithmConfigurationCount;
          auto &candidate = candidates[static_cast<size_t>(configuration)];
          const auto begin = std::chrono::steady_clock::now();
          candidate(a, b, nullptr, 1, 0, workload.trans_a, workload.trans_b, output);
          const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - begin)
                              .count();
          const auto &tuning = candidate.tuning();
          std::cout << dtype << ',' << workload.name << ',' << workload.m << ',' << workload.n
                    << ',' << workload.k << ',' << workload.trans_a << ',' << workload.trans_b
                    << ',' << configuration << ',' << tuning.tile_m << ',' << tuning.tile_n << ','
                    << tuning.tile_k << ',' << tuning.pack_b_minimum_elements << ',' << repetition
                    << ',' << ns << '\n';
        }
      }
    }
  }
}

int Calibrate(const CpuExecutionDescriptor &execution, const char *path) {
  KernelCalibrationSelection selection;
  selection.kernels = {"Gemm"};
  CalibrationOptions options;
  options.execution = execution;
  options.parameter_name = onnx_kernels::tuning::kGemmAlgorithmConfiguration;
  options.maximum_duration_ms = 2000;
  options.maximum_memory_bytes = 8 << 20;
  const auto report = CalibrateRegisteredKernels(selection, options);
  for (const auto &diagnostic : report.diagnostics) {
    std::cout << "diagnostic," << diagnostic.key.element_type << ','
              << std::quoted(diagnostic.message) << '\n';
  }
  for (const auto &resource : report.resources) {
    std::cout << "resource," << resource.key.element_type << ',' << resource.benchmark_cases << ','
              << resource.peak_memory_bytes << ',' << resource.measured_duration_ns << '\n';
  }
  for (const auto &comparison : report.comparisons) {
    for (const auto &value : comparison.comparison.values) {
      std::cout << "comparison," << comparison.key.element_type << ',' << value.value << ','
                << value.duration_ns << ',' << value.benchmark_cases << '\n';
    }
  }
  KernelTuningCacheOptions cache;
  cache.path = path;
  cache.execution = execution;
  const auto update = UpdateKernelTuningCache(report.calibrated, cache);
  std::cout << "persisted," << update.updated.size() << '\n';
  return update.status == KernelTuningCacheUpdateStatus::kUpdated && update.updated.size() == 4 ? 0
                                                                                                : 2;
}

int Reload(const CpuExecutionDescriptor &execution, const char *path, bool incompatible) {
  KernelCalibrationSelection selection;
  selection.kernels = {"Gemm"};
  KernelTuningCacheOptions cache;
  cache.path = path;
  cache.execution = execution;
  if (incompatible) {
    cache.execution->processor.model = execution.processor.model.value_or(0) + 1;
  }
  const auto report = LoadKernelTuningCache(selection, cache);
  for (const auto &diagnostic : report.diagnostics) {
    std::cout << "diagnostic," << std::quoted(diagnostic) << '\n';
  }
  size_t resolved = 0;
  const auto snapshot = GetKernelTuningRegistry().Snapshot();
  for (int32_t dtype : kTypes) {
    const onnx_kernels::kernel::Gemm gemm{KernelContext{DefaultOpset(13)}};
    const auto key = gemm.TuningKey(dtype);
    if (snapshot.HasPublishedProfile(key, *cache.execution)) {
      ++resolved;
      const auto *parameters = snapshot.Resolve(key, *cache.execution);
      std::cout << "selected," << dtype << ','
                << parameters->Get<int64_t>(onnx_kernels::tuning::kGemmAlgorithmConfiguration)
                << '\n';
    }
  }
  std::cout << "loaded," << report.loaded.size() << "\nincompatible," << report.incompatible.size()
            << "\nresolved," << resolved << '\n';
  return incompatible ? (report.incompatible.size() == 4 && resolved == 0 ? 0 : 2)
                      : (report.loaded.size() == 4 && resolved == 4 ? 0 : 2);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4 || argc > 5) {
    std::cerr << "Usage: bench_gemm_calibration measure|calibrate|reload|incompatible "
                 "threads revision [cache_path]\n";
    return 1;
  }
  const std::string_view mode(argv[1]);
  const std::string_view threads_text(argv[2]);
  int threads = 0;
  const auto parsed =
      std::from_chars(threads_text.data(), threads_text.data() + threads_text.size(), threads);
  if (parsed.ec != std::errc{} || parsed.ptr != threads_text.data() + threads_text.size() ||
      threads < 1 || threads > 256 ||
      (mode != "measure" && mode != "calibrate" && mode != "reload" && mode != "incompatible") ||
      (mode != "measure" && argc != 5)) {
    std::cerr << "Invalid mode, thread count, or missing cache path.\n";
    return 1;
  }
  onnx_kernels::kernel::Gemm::RegisterTuningSchemas();
  RuntimeSessionOptions options;
  CpuExecutionPolicy policy;
  policy.num_threads = threads;
  policy.affinity_policy = CpuAffinityPolicy::kNone;
  options.cpu_execution = policy;
  RuntimeSession session(ExecutionPlan{}, options);
  const auto &executor = session.cpu_executor();
  const CpuExecutorScope scope(executor.get());
  const CpuExecutionDescriptor execution{core::platform::GetCpuDescriptor(),
                                         executor->policy().effective_threads};
  PrintDescriptor(execution, executor->policy(), argv[3]);
  if (mode == "measure") {
    Measure();
    return 0;
  }
  return mode == "calibrate" ? Calibrate(execution, argv[4])
                             : Reload(execution, argv[4], mode == "incompatible");
}
