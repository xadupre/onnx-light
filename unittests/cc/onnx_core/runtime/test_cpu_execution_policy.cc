// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/cpu_execution_policy.h"

#include "onnx_core/platform/cpu_descriptor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <thread>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

// Resolves the topology-derived default participant count the same way the
// implementation does, so tests do not hard-code a value that depends on the
// host.
uint32_t ExpectedDefaultThreads() {
  const platform::CpuDescriptor &descriptor = platform::GetCpuDescriptor();
  if (descriptor.physical_cores.has_value() && *descriptor.physical_cores != 0) {
    return *descriptor.physical_cores;
  }
  if (descriptor.logical_cores.has_value() && *descriptor.logical_cores != 0) {
    return *descriptor.logical_cores;
  }
  const unsigned int hardware = std::thread::hardware_concurrency();
  return hardware == 0 ? 1 : static_cast<uint32_t>(hardware);
}

TEST(CpuExecutionPolicy, DefaultResolvesToTopologyThreads) {
  CpuExecutionPolicy request;
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_EQ(resolved.effective_threads, ExpectedDefaultThreads());
  EXPECT_GE(resolved.effective_threads, 1u);
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
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {CpuLogicalProcessor{0}, CpuLogicalProcessor{1}};
  ResolvedCpuExecutionPolicy resolved = ResolveCpuExecutionPolicy(request);
  EXPECT_EQ(resolved.effective_threads, 2u);
  EXPECT_EQ(resolved.worker_processors, request.cpu_set);
}

TEST(CpuExecutionPolicy, ExplicitAffinityRequiresCpuSet) {
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, ExplicitAffinityRejectsDuplicates) {
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {CpuLogicalProcessor{0}, CpuLogicalProcessor{0}};
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, ExplicitAffinityRejectsThreadCountMismatch) {
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {CpuLogicalProcessor{0}, CpuLogicalProcessor{1}};
  request.num_threads = 3;
  EXPECT_THROW(ResolveCpuExecutionPolicy(request), std::invalid_argument);
}

TEST(CpuExecutionPolicy, ExplicitAffinityRejectsSerialWithMultipleProcessors) {
  CpuExecutionPolicy request;
  request.affinity_policy = CpuAffinityPolicy::kExplicit;
  request.cpu_set = {CpuLogicalProcessor{0}, CpuLogicalProcessor{1}};
  request.num_threads = 1;
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
  request.num_threads = 4;
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
