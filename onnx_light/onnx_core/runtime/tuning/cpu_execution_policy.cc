// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/cpu_execution_policy.h"

#include "onnx_core/platform/cpu_descriptor.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

std::optional<uint32_t> ReadUnsignedFile(const std::filesystem::path &path) {
  std::ifstream input(path);
  uint32_t value = 0;
  if (!(input >> value)) {
    return std::nullopt;
  }
  return value;
}

struct LogicalProcessorTopology {
  CpuLogicalProcessor processor;
  uint32_t package_id = 0;
  uint32_t core_id = 0;
};

std::vector<LogicalProcessorTopology>
ReadProcessorTopology(const std::vector<CpuLogicalProcessor> &processors) {
  std::vector<LogicalProcessorTopology> topology;
#if defined(__linux__)
  topology.reserve(processors.size());
  for (const CpuLogicalProcessor &processor : processors) {
    const std::filesystem::path path = std::filesystem::path("/sys/devices/system/cpu") /
                                       ("cpu" + std::to_string(processor.id)) / "topology";
    const std::optional<uint32_t> core_id = ReadUnsignedFile(path / "core_id");
    const std::optional<uint32_t> package_id = ReadUnsignedFile(path / "physical_package_id");
    if (!core_id.has_value() || !package_id.has_value()) {
      return {};
    }
    topology.push_back(LogicalProcessorTopology{processor, *package_id, *core_id});
  }
#else
  (void)processors;
#endif
  return topology;
}

std::optional<uint32_t>
PhysicalCoreCount(const std::vector<CpuLogicalProcessor> &processors) noexcept {
  const std::vector<LogicalProcessorTopology> topology = ReadProcessorTopology(processors);
  if (!topology.empty()) {
    std::set<std::pair<uint32_t, uint32_t>> cores;
    for (const LogicalProcessorTopology &entry : topology) {
      cores.emplace(entry.package_id, entry.core_id);
    }
    return static_cast<uint32_t>(cores.size());
  }
  return std::nullopt;
}

uint32_t
DetectedPhysicalCores(const std::vector<CpuLogicalProcessor> &visible_processors) noexcept {
  const std::optional<uint32_t> visible_physical = PhysicalCoreCount(visible_processors);
  if (visible_physical.has_value()) {
    return *visible_physical;
  }
  const platform::CpuDescriptor &descriptor = platform::GetCpuDescriptor();
  if (descriptor.physical_cores.has_value() && *descriptor.physical_cores != 0) {
    if (!visible_processors.empty()) {
      return std::min(*descriptor.physical_cores, static_cast<uint32_t>(visible_processors.size()));
    }
    return *descriptor.physical_cores;
  }
  return 0;
}

uint32_t DetectedLogicalCores(const std::vector<CpuLogicalProcessor> &visible_processors) noexcept {
  if (!visible_processors.empty()) {
    return static_cast<uint32_t>(visible_processors.size());
  }
  const platform::CpuDescriptor &descriptor = platform::GetCpuDescriptor();
  if (descriptor.logical_cores.has_value() && *descriptor.logical_cores != 0) {
    return *descriptor.logical_cores;
  }
  const unsigned int hardware = std::thread::hardware_concurrency();
  return hardware == 0 ? 0 : static_cast<uint32_t>(hardware);
}

// Resolves the topology-derived default participant count, recording a
// diagnostic when the physical-core count is unavailable and a fallback is used.
uint32_t ResolveDefaultThreads(const std::vector<CpuLogicalProcessor> &visible_processors,
                               std::vector<std::string> &diagnostics) {
  if (visible_processors.empty()) {
    diagnostics.emplace_back(
        "stable process-visible logical processor identifiers unavailable; using processor "
        "descriptor counts");
  }
  const uint32_t physical = DetectedPhysicalCores(visible_processors);
  if (physical != 0) {
    return physical;
  }
  const uint32_t logical = DetectedLogicalCores(visible_processors);
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

void ValidateCpuSet(const CpuExecutionPolicy &request,
                    const std::vector<CpuLogicalProcessor> &visible_processors) {
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
  if (visible_processors.empty()) {
    throw std::invalid_argument(
        "CpuExecutionPolicy explicit affinity is unsupported because stable process-visible "
        "logical processor identifiers are unavailable.");
  }
  std::vector<uint32_t> ids;
  ids.reserve(request.cpu_set.size());
  for (const CpuLogicalProcessor &processor : request.cpu_set) {
    if (std::find(visible_processors.begin(), visible_processors.end(), processor) ==
        visible_processors.end()) {
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

void ValidateParticipantCount(const CpuExecutionPolicy &request, uint32_t effective_threads,
                              uint32_t physical_cores, uint32_t logical_processors) {
  if (request.affinity_policy == CpuAffinityPolicy::kNone ||
      request.affinity_policy == CpuAffinityPolicy::kExplicit) {
    return;
  }
  uint32_t maximum = logical_processors;
  if ((request.affinity_policy == CpuAffinityPolicy::kPhysicalCores ||
       request.affinity_policy == CpuAffinityPolicy::kPerformanceCores) &&
      physical_cores != 0) {
    maximum = physical_cores;
  }
  if (maximum != 0 && effective_threads > maximum) {
    throw std::invalid_argument(
        "CpuExecutionPolicy num_threads exceeds the processors available to the requested "
        "affinity policy.");
  }
}

std::vector<CpuLogicalProcessor>
ResolveTopologyWorkers(const CpuExecutionPolicy &request,
                       const std::vector<CpuLogicalProcessor> &visible_processors,
                       uint32_t effective_threads) {
  const std::vector<LogicalProcessorTopology> topology = ReadProcessorTopology(visible_processors);
  if (topology.empty() || effective_threads <= 1) {
    return {};
  }

  std::optional<std::pair<uint32_t, uint32_t>> caller_core;
  std::optional<CpuLogicalProcessor> caller_processor;
#if defined(__linux__)
  const int caller_id = sched_getcpu();
  if (caller_id >= 0) {
    auto found = std::find_if(topology.begin(), topology.end(),
                              [caller_id](const LogicalProcessorTopology &entry) {
                                return entry.processor.id == static_cast<uint32_t>(caller_id);
                              });
    if (found != topology.end()) {
      caller_processor = found->processor;
      caller_core = std::pair<uint32_t, uint32_t>{found->package_id, found->core_id};
    }
  }
#endif

  std::vector<CpuLogicalProcessor> primary;
  std::vector<CpuLogicalProcessor> siblings;
  std::set<std::pair<uint32_t, uint32_t>> selected_cores;
  if (caller_core.has_value()) {
    selected_cores.insert(*caller_core);
  }
  for (const LogicalProcessorTopology &entry : topology) {
    if (caller_processor.has_value() && entry.processor == *caller_processor) {
      continue;
    }
    const std::pair<uint32_t, uint32_t> core{entry.package_id, entry.core_id};
    if (selected_cores.insert(core).second) {
      primary.push_back(entry.processor);
    } else {
      siblings.push_back(entry.processor);
    }
  }

  std::vector<CpuLogicalProcessor> workers = std::move(primary);
  if (request.affinity_policy == CpuAffinityPolicy::kPhysicalThenSmt) {
    workers.insert(workers.end(), siblings.begin(), siblings.end());
  }
  const size_t worker_count = static_cast<size_t>(effective_threads - 1);
  if (workers.size() < worker_count) {
    return {};
  }
  workers.resize(worker_count);
  return workers;
}

} // namespace

std::vector<CpuLogicalProcessor> ProcessVisibleLogicalProcessors() {
  std::vector<CpuLogicalProcessor> processors;
#if defined(__linux__)
  const long configured = sysconf(_SC_NPROCESSORS_CONF);
  if (configured <= 0) {
    return processors;
  }
  cpu_set_t *affinity = CPU_ALLOC(static_cast<size_t>(configured));
  if (affinity == nullptr) {
    return processors;
  }
  const size_t affinity_size = CPU_ALLOC_SIZE(static_cast<size_t>(configured));
  CPU_ZERO_S(affinity_size, affinity);
  if (sched_getaffinity(0, affinity_size, affinity) == 0) {
    for (uint32_t id = 0; id < static_cast<uint32_t>(configured); ++id) {
      if (CPU_ISSET_S(static_cast<int>(id), affinity_size, affinity)) {
        processors.push_back(CpuLogicalProcessor{id});
      }
    }
  }
  CPU_FREE(affinity);
#elif defined(_WIN32)
  USHORT group_capacity = GetActiveProcessorGroupCount();
  std::vector<USHORT> groups(group_capacity);
  if (group_capacity == 0 ||
      GetProcessGroupAffinity(GetCurrentProcess(), &group_capacity, groups.data()) == 0 ||
      group_capacity != 1) {
    return processors;
  }
  DWORD_PTR process_mask = 0;
  DWORD_PTR system_mask = 0;
  if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) == 0) {
    return processors;
  }
  for (uint32_t id = 0; id < sizeof(DWORD_PTR) * 8; ++id) {
    if ((process_mask & (static_cast<DWORD_PTR>(1) << id)) != 0) {
      processors.push_back(CpuLogicalProcessor{id, groups[0]});
    }
  }
#endif
  return processors;
}

ResolvedCpuExecutionPolicy ResolveCpuExecutionPolicy(const CpuExecutionPolicy &request) {
  if (request.num_threads < 0) {
    throw std::invalid_argument("CpuExecutionPolicy num_threads must not be negative.");
  }
  const std::vector<CpuLogicalProcessor> visible_processors = ProcessVisibleLogicalProcessors();
  ValidateSpinPolicy(request);
  ValidateCpuSet(request, visible_processors);

  ResolvedCpuExecutionPolicy resolved;
  resolved.request = request;
  resolved.allow_nested_parallelism = request.allow_nested_parallelism;
  resolved.spin = ResolveSpin(request);

  const bool explicit_affinity = request.affinity_policy == CpuAffinityPolicy::kExplicit;
  if (explicit_affinity && request.num_threads > 0 &&
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
    resolved.effective_threads = ResolveDefaultThreads(visible_processors, resolved.diagnostics);
  }

  const uint32_t physical = DetectedPhysicalCores(visible_processors);
  const uint32_t logical = DetectedLogicalCores(visible_processors);
  ValidateParticipantCount(request, resolved.effective_threads, physical, logical);
  if (explicit_affinity) {
    const std::optional<uint32_t> selected_physical = PhysicalCoreCount(request.cpu_set);
    resolved.uses_smt =
        selected_physical.has_value() && request.cpu_set.size() > *selected_physical;
  } else {
    resolved.uses_smt = physical != 0 && resolved.effective_threads > physical;
  }

  if (request.affinity_policy == CpuAffinityPolicy::kNone) {
    return resolved;
  }

  if (explicit_affinity) {
    resolved.caller_processor = request.cpu_set.front();
    resolved.worker_processors.assign(request.cpu_set.begin() + 1, request.cpu_set.end());
    return resolved;
  }

  if (request.affinity_policy == CpuAffinityPolicy::kPerformanceCores) {
    resolved.diagnostics.emplace_back(
        "performance/efficiency core identification unavailable; treating all cores as equal");
  }

  if (resolved.effective_threads <= 1) {
    return resolved;
  }

  resolved.worker_processors =
      ResolveTopologyWorkers(request, visible_processors, resolved.effective_threads);
  if (resolved.worker_processors.size() == static_cast<size_t>(resolved.effective_threads - 1)) {
    return resolved;
  }

  resolved.diagnostics.emplace_back(
      "stable per-core processor topology unavailable; workers run without explicit pinning");
  return resolved;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
