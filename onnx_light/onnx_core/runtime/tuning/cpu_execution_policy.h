// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

/**
 * @file cpu_execution_policy.h
 * @brief Requested and resolved CPU execution policy for a runtime session.
 *
 * This is the first step (Pool PR01) of the
 * :ref:`l-next-steps-session-execution-pools` roadmap. It defines the typed
 * policy a session requests and the immutable resolution derived from it. The
 * executor and compatible-pool registry are delivered by Pool PR02. Session
 * wiring is delivered by a later step; this header owns the policy vocabulary
 * and its deterministic validation.
 */

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Identifies a logical processor by a stable operating-system identifier.
 *
 * Identifiers must come from the process-visible CPU set and are never inferred
 * from adjacency. The wrapper keeps the vocabulary extensible without changing
 * every call site.
 */
struct CpuLogicalProcessor {
  /// Stable operating-system logical processor identifier.
  uint32_t id = 0;
  /// Windows processor group. It is ``0`` on platforms without processor
  /// groups.
  uint16_t group = 0;

  bool operator==(const CpuLogicalProcessor &) const = default;
};

/**
 * Controls how a waiting worker or caller spins before parking.
 *
 * Spinning applies both to workers waiting for a new generation and to the
 * caller waiting for worker completion. The default bounds the spin and
 * eventually parks.
 */
enum class CpuSpinPolicy {
  /// Bounded, topology-derived spin that eventually parks.
  kAdaptive,
  /// Spin for an explicit number of iterations before parking.
  kFixedIterations,
  /// Spin for an explicit duration in nanoseconds before parking.
  kFixedDuration,
  /// Park immediately without spinning.
  kParkImmediately,
};

/**
 * Controls how workers are placed on the process-visible logical processors.
 */
enum class CpuAffinityPolicy {
  /// Do not pin workers; a successful no-affinity policy.
  kNone,
  /// One logical processor per physical core.
  kPhysicalCores,
  /// Prefer performance cores when the topology can identify them.
  kPerformanceCores,
  /// One logical processor per physical core, then SMT siblings.
  kPhysicalThenSmt,
  /// Pin workers to an explicit CPU set.
  kExplicit,
};

/**
 * Requested per-session CPU execution policy.
 *
 * Names are illustrative and may change during API review. The policy is
 * resolved into an immutable :cpp:class:`ResolvedCpuExecutionPolicy` by
 * :cpp:func:`ResolveCpuExecutionPolicy`.
 */
struct CpuExecutionPolicy {
  /** Requested number of participants, including the calling thread.
   *  - ``0`` (default): use a topology-derived default.
   *  - ``1``: serial, no worker threads.
   *  - ``> 1``: request exactly this many participants.
   *  - ``< 0``: rejected. */
  int32_t num_threads = 0;
  /// Requested spin-before-park policy.
  CpuSpinPolicy spin_policy = CpuSpinPolicy::kAdaptive;
  /// Iterations for :cpp:enumerator:`CpuSpinPolicy::kFixedIterations` or
  /// nanoseconds for :cpp:enumerator:`CpuSpinPolicy::kFixedDuration`. Must be
  /// ``0`` for the adaptive and park-immediately policies.
  uint64_t spin_budget = 0;
  /// Requested affinity policy.
  CpuAffinityPolicy affinity_policy = CpuAffinityPolicy::kNone;
  /// Explicit participant CPU set, required and only allowed for
  /// :cpp:enumerator:`CpuAffinityPolicy::kExplicit`. The first processor is
  /// assigned to the calling participant; the remaining processors are
  /// assigned to workers.
  std::vector<CpuLogicalProcessor> cpu_set;
  /// Whether nested parallel regions may create additional participants.
  bool allow_nested_parallelism = false;

  bool operator==(const CpuExecutionPolicy &) const = default;
};

/**
 * Immutable resolution of a spin policy.
 *
 * Exactly one of :cpp:var:`iterations` and :cpp:var:`duration_ns` is non-zero
 * for the fixed policies; both are ``0`` for the park-immediately policy.
 */
struct ResolvedSpinPolicy {
  /// Resolved spin policy class.
  CpuSpinPolicy policy = CpuSpinPolicy::kAdaptive;
  /// Resolved spin iterations before parking, or ``0`` when not iteration-based.
  uint64_t iterations = 0;
  /// Resolved spin duration in nanoseconds before parking, or ``0`` when not
  /// duration-based.
  uint64_t duration_ns = 0;

  bool operator==(const ResolvedSpinPolicy &) const = default;
};

/**
 * Immutable resolution of a :cpp:class:`CpuExecutionPolicy`.
 *
 * The resolution describes the workers that actually execute the graph. It is
 * derived deterministically from the request and the process-visible topology.
 */
struct ResolvedCpuExecutionPolicy {
  /// The request this resolution was derived from.
  CpuExecutionPolicy request;
  /// Effective number of participants, including the calling thread. Always
  /// ``>= 1``.
  uint32_t effective_threads = 1;
  /// Explicit calling-participant assignment, absent when the caller is not
  /// pinned by this policy.
  std::optional<CpuLogicalProcessor> caller_processor;
  /// Explicit worker processor assignment, empty when no pinning is applied.
  /// Its size is at most ``effective_threads - 1``.
  std::vector<CpuLogicalProcessor> worker_processors;
  /// Whether the resolution relies on SMT siblings.
  bool uses_smt = false;
  /// Whether the resolution relies on efficiency cores.
  bool uses_efficiency_cores = false;
  /// Resolved spin and park policy.
  ResolvedSpinPolicy spin;
  /// Whether nested parallel regions may create additional participants.
  bool allow_nested_parallelism = false;
  /// Human-readable notes about fallbacks taken during resolution.
  std::vector<std::string> diagnostics;

  bool operator==(const ResolvedCpuExecutionPolicy &) const = default;
};

/// Default number of adaptive spin iterations recorded before parking.
inline constexpr uint64_t kDefaultAdaptiveSpinIterations = 10000;

/**
 * Returns stable identifiers for the logical processors currently available
 * to this process.
 *
 * The returned set reflects process affinity restrictions such as Linux
 * cpusets. It is empty when the operating system cannot expose stable logical
 * processor identifiers.
 *
 * Returns:
 *   The process-visible logical processors in increasing identifier order.
 */
std::vector<CpuLogicalProcessor> ProcessVisibleLogicalProcessors();

/**
 * Returns the detected physical core count for the process-visible topology.
 *
 * Falls back to the platform CPU descriptor when the topology cannot be read,
 * matching the detection :cpp:func:`ResolveCpuExecutionPolicy` uses to bound
 * :cpp:enumerator:`CpuAffinityPolicy::kPhysicalCores` participant counts.
 *
 * Returns:
 *   The detected physical core count, or ``0`` when it cannot be determined.
 */
uint32_t DetectedPhysicalCoreCount() noexcept;

/**
 * Resolves a requested CPU execution policy into an immutable resolution.
 *
 * Resolution validates the request deterministically and derives the effective
 * participant count and worker placement from the process-visible topology.
 * Fallbacks taken when a requested topology feature is unavailable are recorded
 * in :cpp:var:`ResolvedCpuExecutionPolicy::diagnostics` rather than failing.
 *
 * The following requests fail with :cpp:class:`std::invalid_argument`:
 *   - a negative :cpp:var:`CpuExecutionPolicy::num_threads`;
 *   - a spin budget that is zero for a fixed policy or non-zero for the
 *     adaptive or park-immediately policy;
 *   - a :cpp:enumerator:`CpuAffinityPolicy::kExplicit` policy with an empty CPU
 *     set, duplicate identifiers, or an identifier outside the process-visible
 *     set;
 *   - a non-explicit affinity policy that supplies a CPU set;
 *   - an explicit CPU set whose size conflicts with a positive requested thread
 *     count.
 *
 * @param request The requested policy.
 *
 * Returns:
 *   The immutable resolution derived from the request.
 */
ResolvedCpuExecutionPolicy ResolveCpuExecutionPolicy(const CpuExecutionPolicy &request);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
