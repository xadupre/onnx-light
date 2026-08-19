// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/cpu_execution_policy.h"

#include "onnx_core/platform/cpu_descriptor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

#if defined(__linux__)
std::optional<std::pair<uint32_t, uint32_t>> ProcessorCore(CpuLogicalProcessor processor) {
  const std::filesystem::path topology = std::filesystem::path("/sys/devices/system/cpu") /
                                         ("cpu" + std::to_string(processor.id)) / "topology";
  std::ifstream package_input(topology / "physical_package_id");
  std::ifstream core_input(topology / "core_id");
  uint32_t package_id = 0;
  uint32_t core_id = 0;
  if (!(package_input >> package_id) || !(core_input >> core_id)) {
    return std::nullopt;
  }
  return std::pair<uint32_t, uint32_t>{package_id, core_id};
}
#endif

TEST(CpuExecutionPolicy, ProcessVisibleProcessorsAreStableAndUnique) {
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  EXPECT_TRUE(std::is_sorted(visible.begin(), visible.end(),
                             [](const CpuLogicalProcessor &left, const CpuLogicalProcessor &right) {
                               return left.id < right.id;
                             }));
  EXPECT_EQ(std::adjacent_find(visible.begin(), visible.end()), visible.end());
}

TEST(CpuExecutionPolicy, DefaultResolvesToTopologyThreads) {
  CpuExecutionPolicy request;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_GE(resolved.effective_threads, 1u);
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  if (!visible.empty()) {
    EXPECT_LE(resolved.effective_threads, visible.size());
  }
  EXPECT_EQ(resolved.request, request);
  EXPECT_EQ(resolved.spin.policy, CpuSpinPolicy::kAdaptive);
  EXPECT_EQ(resolved.spin.iterations, kDefaultAdaptiveSpinIterations);
}

TEST(CpuExecutionPolicy, SerialHasNoWorkerProcessors) {
  CpuExecutionPolicy request;
  request.num_threads = 1;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_EQ(resolved.effective_threads, 1u);
  EXPECT_FALSE(resolved.uses_smt);
  EXPECT_TRUE(resolved.worker_processors.empty());
}

TEST(CpuExecutionPolicy, ExplicitThreadCountIsHonored) {
  CpuExecutionPolicy request;
  request.num_threads = 5;
  request.affinity_policy = CpuAffinityPolicy::kNone;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_EQ(resolved.effective_threads, 5u);
  EXPECT_TRUE(resolved.worker_processors.empty());
}

TEST(CpuExecutionPolicy, NegativeThreadCountThrows) {
  CpuExecutionPolicy request;
  request.num_threads = -1;
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, FixedIterationsSpinBudgetResolves) {
  CpuExecutionPolicy request;
  request.spin_policy = CpuSpinPolicy::kFixedIterations;
  request.spin_budget = 4096;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_EQ(resolved.spin.policy, CpuSpinPolicy::kFixedIterations);
  EXPECT_EQ(resolved.spin.iterations, 4096u);
  EXPECT_EQ(resolved.spin.duration_ns, 0u);
}

TEST(CpuExecutionPolicy, FixedDurationSpinBudgetResolves) {
  CpuExecutionPolicy request;
  request.spin_policy = CpuSpinPolicy::kFixedDuration;
  request.spin_budget = 250000;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_EQ(resolved.spin.policy, CpuSpinPolicy::kFixedDuration);
  EXPECT_EQ(resolved.spin.duration_ns, 250000u);
  EXPECT_EQ(resolved.spin.iterations, 0u);
}

TEST(CpuExecutionPolicy, ParkImmediatelyHasNoSpin) {
  CpuExecutionPolicy request;
  request.spin_policy = CpuSpinPolicy::kParkImmediately;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_EQ(resolved.spin.iterations, 0u);
  EXPECT_EQ(resolved.spin.duration_ns, 0u);
}

TEST(CpuExecutionPolicy, FixedSpinRequiresPositiveBudget) {
  CpuExecutionPolicy request;
  request.spin_policy = CpuSpinPolicy::kFixedIterations;
  request.spin_budget = 0;
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, AdaptiveSpinRejectsBudget) {
  CpuExecutionPolicy request;
  request.spin_policy = CpuSpinPolicy::kAdaptive;
  request.spin_budget = 10;
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, ExplicitAffinityPinsWorkers) {
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  if (visible.size() < 2) {
    GTEST_SKIP() << "fewer than two stable process-visible processor identifiers";
  }
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {visible[0], visible[1]};
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_EQ(resolved.effective_threads, 2u);
  EXPECT_EQ(resolved.caller_processor, request.cpu_set[0]);
  EXPECT_EQ(resolved.worker_processors, std::vector<CpuLogicalProcessor>({request.cpu_set[1]}));
}

TEST(CpuExecutionPolicy, ExplicitSerialRetainsCallerProcessor) {
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  if (visible.empty()) {
    GTEST_SKIP() << "stable process-visible processor identifiers unavailable";
  }
  CpuExecutionPolicy request;
  request.num_threads = 1;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {visible[0]};
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_EQ(resolved.effective_threads, 1u);
  EXPECT_EQ(resolved.caller_processor, visible[0]);
  EXPECT_TRUE(resolved.worker_processors.empty());
}

TEST(CpuExecutionPolicy, ExplicitAffinityRequiresCpuSet) {
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, ExplicitAffinityRejectsDuplicates) {
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  if (visible.empty()) {
    GTEST_SKIP() << "stable process-visible processor identifiers unavailable";
  }
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {visible[0], visible[0]};
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, ExplicitAffinityRejectsThreadCountMismatch) {
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  if (visible.size() < 2) {
    GTEST_SKIP() << "fewer than two stable process-visible processor identifiers";
  }
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {visible[0], visible[1]};
  request.num_threads = 3;
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, ExplicitAffinityRejectsSerialWithMultipleProcessors) {
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  if (visible.size() < 2) {
    GTEST_SKIP() << "fewer than two stable process-visible processor identifiers";
  }
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {visible[0], visible[1]};
  request.num_threads = 1;
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, ExplicitAffinityRejectsProcessorOutsideVisibleSet) {
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  if (visible.empty()) {
    GTEST_SKIP() << "stable process-visible processor identifiers unavailable";
  }
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {CpuLogicalProcessor{visible.back().id + 1}};
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, NonExplicitAffinityRejectsCpuSet) {
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kPhysicalCores;
  request.cpu_set = {CpuLogicalProcessor{0}};
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, PerformanceCoresRecordsFallbackDiagnostic) {
  CpuExecutionPolicy request;
  request.num_threads = 1;
  request.affinity_policy = CpuAffinityPolicy::kPerformanceCores;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  const bool has_pe_diagnostic = std::any_of(
      resolved.diagnostics.begin(), resolved.diagnostics.end(), [](const std::string &message) {
        return message.find("performance/efficiency") != std::string::npos;
      });
  EXPECT_TRUE(has_pe_diagnostic);
  EXPECT_TRUE(resolved.worker_processors.empty());
}

TEST(CpuExecutionPolicy, NestingFlagIsCarried) {
  CpuExecutionPolicy request;
  request.allow_nested_parallelism = true;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_TRUE(resolved.allow_nested_parallelism);
}

TEST(CpuExecutionPolicy, PhysicalThenSmtRejectsMoreThanVisibleProcessors) {
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  if (visible.empty()) {
    GTEST_SKIP() << "process-visible logical processor count unavailable";
  }
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kPhysicalThenSmt;
  request.num_threads = static_cast<int32_t>(visible.size()) + 1;
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, PhysicalCoreAffinityAssignsEveryWorker) {
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kPhysicalCores;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  if (resolved.effective_threads <= 1) {
    GTEST_SKIP() << "fewer than two process-visible physical cores";
  }
  EXPECT_EQ(resolved.worker_processors.size(), static_cast<size_t>(resolved.effective_threads - 1));
#if defined(__linux__)
  std::set<std::pair<uint32_t, uint32_t>> worker_cores;
  for (CpuLogicalProcessor processor : resolved.worker_processors) {
    const std::optional<std::pair<uint32_t, uint32_t>> core = ProcessorCore(processor);
    ASSERT_TRUE(core.has_value());
    EXPECT_TRUE(worker_cores.insert(*core).second);
  }
#endif
}

TEST(CpuExecutionPolicy, PhysicalThenSmtPlacesSiblingsAfterPhysicalCores) {
#if defined(__linux__)
  const std::vector<CpuLogicalProcessor> visible = ProcessVisibleLogicalProcessors();
  std::set<std::pair<uint32_t, uint32_t>> visible_cores;
  for (CpuLogicalProcessor processor : visible) {
    const std::optional<std::pair<uint32_t, uint32_t>> core = ProcessorCore(processor);
    if (!core.has_value()) {
      GTEST_SKIP() << "process-visible core topology unavailable";
    }
    visible_cores.insert(*core);
  }
  if (visible.size() <= visible_cores.size()) {
    GTEST_SKIP() << "no process-visible SMT siblings";
  }

  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kPhysicalThenSmt;
  request.num_threads = static_cast<int32_t>(visible.size());
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  ASSERT_EQ(resolved.worker_processors.size(), visible.size() - 1);
  EXPECT_TRUE(resolved.uses_smt);

  std::set<std::pair<uint32_t, uint32_t>> primary_cores;
  const size_t primary_workers = visible_cores.size() - 1;
  for (size_t index = 0; index < primary_workers; ++index) {
    const std::optional<std::pair<uint32_t, uint32_t>> core =
        ProcessorCore(resolved.worker_processors[index]);
    ASSERT_TRUE(core.has_value());
    EXPECT_TRUE(primary_cores.insert(*core).second);
  }
#else
  GTEST_SKIP() << "stable per-core topology unavailable";
#endif
}

TEST(CpuExecutionPolicy, SmtIsReportedWhenExceedingPhysicalCores) {
  const platform::CpuDescriptor &descriptor = platform::GetCpuDescriptor();
  if (!descriptor.physical_cores.has_value() || *descriptor.physical_cores == 0) {
    GTEST_SKIP() << "physical core count unavailable on this host";
  }
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kNone;
  request.num_threads = static_cast<int32_t>(*descriptor.physical_cores) + 1;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_TRUE(resolved.uses_smt);
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
