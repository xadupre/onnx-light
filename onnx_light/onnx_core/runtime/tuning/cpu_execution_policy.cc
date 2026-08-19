// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/cpu_execution_policy.h"

#include "onnx_core/platform/cpu_descriptor.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

// Returns the number of physical cores visible to the process, or 0 when the
// operating system cannot report it.
uint32_t DetectedPhysicalCores() noexcept {
  const platform::CpuDescriptor &descriptor = platform::GetCpuDescriptor();
  if (descriptor.physical_cores.has_value() && *descriptor.physical_cores != 0) {
    return *descriptor.physical_cores;
  }
  return 0;
}

// Returns the number of logical processors visible to the process, or 0 when
// the operating system cannot report it.
uint32_t DetectedLogicalCores() noexcept {
  const platform::CpuDescriptor &descriptor = platform::GetCpuDescriptor();
  if (descriptor.logical_cores.has_value() && *descriptor.logical_cores != 0) {
    return *descriptor.logical_cores;
  }
  const unsigned int hardware = std::thread::hardware_concurrency();
  return hardware == 0 ? 0 : static_cast<uint32_t>(hardware);
}

// Resolves the topology-derived default participant count, recording a
// diagnostic when the physical-core count is unavailable and a fallback is used.
uint32_t ResolveDefaultThreads(std::vector<std::string> &diagnostics) {
  const uint32_t physical = DetectedPhysicalCores();
  if (physical != 0) {
    return physical;
  }
  const uint32_t logical = DetectedLogicalCores();
  if (logical != 0) {
    diagnostics.emplace_back(
        "physical core count unavailable; using logical core count for the default thread count");
    return logical;
  }
  diagnostics.emplace_back(
      "processor topology unavailable; defaulting to a single execution thread");
  return 1;
}

void ValidateSpinPolicy(const CpuExecutionPolicy &request) {
  switch (request.spin_policy) {
  case CpuSpinPolicy::kFixedIterations:
  case CpuSpinPolicy::kFixedDuration:
    if (request.spin_budget == 0) {
      throw std::invalid_argument(
          "CpuExecutionPolicy spin_budget must be positive for a fixed spin policy.");
    }
    break;
  case CpuSpinPolicy::kAdaptive:
  case CpuSpinPolicy::kParkImmediately:
    if (request.spin_budget != 0) {
      throw std::invalid_argument("CpuExecutionPolicy spin_budget must be zero for the adaptive "
                                  "and park-immediately spin policies.");
    }
    break;
  }
}

void ValidateCpuSet(const CpuExecutionPolicy &request) {
  if (request.affinity_policy != CpuAffinityPolicy::kExplicit) {
    if (!request.cpu_set.empty()) {
      throw std::invalid_argument(
          "CpuExecutionPolicy cpu_set is only allowed with the explicit affinity policy.");
    }
    return;
  }
  if (request.cpu_set.empty()) {
    throw std::invalid_argument(
        "CpuExecutionPolicy cpu_set must not be empty for the explicit affinity policy.");
  }
  const uint32_t logical = DetectedLogicalCores();
  std::vector<uint32_t> ids;
  ids.reserve(request.cpu_set.size());
  for (const CpuLogicalProcessor &processor : request.cpu_set) {
    if (logical != 0 && processor.id >= logical) {
      throw std::invalid_argument("CpuExecutionPolicy cpu_set contains a logical processor "
                                  "identifier outside the process-visible set.");
    }
    ids.push_back(processor.id);
  }
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
    throw std::invalid_argument(
        "CpuExecutionPolicy cpu_set must not contain duplicate logical processor identifiers.");
  }
}

ResolvedSpinPolicy ResolveSpin(const CpuExecutionPolicy &request) {
  ResolvedSpinPolicy resolved;
  resolved.policy = request.spin_policy;
  switch (request.spin_policy) {
  case CpuSpinPolicy::kAdaptive:
    resolved.iterations = kDefaultAdaptiveSpinIterations;
    break;
  case CpuSpinPolicy::kFixedIterations:
    resolved.iterations = request.spin_budget;
    break;
  case CpuSpinPolicy::kFixedDuration:
    resolved.duration_ns = request.spin_budget;
    break;
  case CpuSpinPolicy::kParkImmediately:
    break;
  }
  return resolved;
}

} // namespace

ResolvedCpuExecutionPolicy ResolveCpuExecutionPolicy(const CpuExecutionPolicy &request) {
  if (request.num_threads < 0) {
    throw std::invalid_argument("CpuExecutionPolicy num_threads must not be negative.");
  }
  ValidateSpinPolicy(request);
  ValidateCpuSet(request);

  ResolvedCpuExecutionPolicy resolved;
  resolved.request = request;
  resolved.allow_nested_parallelism = request.allow_nested_parallelism;
  resolved.spin = ResolveSpin(request);

  const bool explicit_affinity = request.affinity_policy == CpuAffinityPolicy::kExplicit;
  if (explicit_affinity && request.num_threads > 1 &&
      static_cast<size_t>(request.num_threads) != request.cpu_set.size()) {
    throw std::invalid_argument("CpuExecutionPolicy num_threads must match the explicit cpu_set "
                                "size when both are specified.");
  }

  if (request.num_threads == 1) {
    resolved.effective_threads = 1;
  } else if (request.num_threads > 1) {
    resolved.effective_threads = static_cast<uint32_t>(request.num_threads);
  } else if (explicit_affinity) {
    resolved.effective_threads = static_cast<uint32_t>(request.cpu_set.size());
  } else {
    resolved.effective_threads = ResolveDefaultThreads(resolved.diagnostics);
  }

  const uint32_t physical = DetectedPhysicalCores();
  resolved.uses_smt = physical != 0 && resolved.effective_threads > physical;

  if (resolved.effective_threads <= 1 || request.affinity_policy == CpuAffinityPolicy::kNone) {
    // Serial execution and the no-affinity policy do not pin workers.
    return resolved;
  }

  if (explicit_affinity) {
    resolved.worker_processors = request.cpu_set;
    return resolved;
  }

  // The process descriptor exposes only core counts, not stable per-core
  // identifiers, so a specific worker placement cannot be derived without
  // inferring identifiers from adjacency. Record the fallback instead.
  if (request.affinity_policy == CpuAffinityPolicy::kPerformanceCores) {
    resolved.diagnostics.emplace_back(
        "performance/efficiency core identification unavailable; treating all cores as equal");
  }
  resolved.diagnostics.emplace_back(
      "stable logical processor identifiers unavailable; workers run without explicit pinning");
  return resolved;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
