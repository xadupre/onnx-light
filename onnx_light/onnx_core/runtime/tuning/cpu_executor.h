// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/tuning/cpu_execution_policy.h"
#include "onnx_core/runtime/tuning/parallel_region_collector.h"
#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <source_location>
#include <string_view>
#include <vector>

/**
 * @file cpu_executor.h
 * @brief Shared CPU executor and compatible-pool registry.
 */

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/// Type-erased range callable: ``function(context, begin, end)``.
using ParallelRangeFn = void (*)(void *, int64_t, int64_t);

/// Type-erased block callable used by an external CPU dispatcher.
using CpuParallelBlockFn = void (*)(void *, int64_t);

/**
 * Dispatches indexed blocks through an externally owned worker pool.
 *
 * The callback must synchronously invoke ``block_function(block_context, i)``
 * exactly once for every ``i`` in ``[0, num_blocks)`` before returning.
 */
using CpuParallelDispatchFn = void (*)(void *dispatch_context, int64_t num_blocks,
                                       void *block_context, CpuParallelBlockFn block_function);

/**
 * Identifies every resolved property that changes executor behavior.
 *
 * Request spelling and diagnostics are intentionally excluded. The key records
 * whether no affinity was explicitly requested so that a successful
 * no-affinity policy is not confused with an unsupported affinity fallback.
 */
struct CpuExecutorKey {
  /// Effective participants, including the caller.
  uint32_t effective_threads = 1;
  /// Optional caller placement.
  std::optional<CpuLogicalProcessor> caller_processor;
  /// Exact worker placement, or an empty vector for unpinned workers.
  std::vector<CpuLogicalProcessor> worker_processors;
  /// Resolved spin-before-park behavior.
  ResolvedSpinPolicy spin;
  /// Whether the request explicitly selected no affinity.
  bool explicit_no_affinity = false;
  /// Whether nested parallelism was requested.
  bool allow_nested_parallelism = false;

  bool operator==(const CpuExecutorKey &) const = default;
};

/** Snapshot of optional executor dispatch counters. */
struct CpuExecutorCounters {
  /// Number of ParallelFor dispatches, including inline dispatches.
  uint64_t dispatches = 0;
  /// Number of dispatches that ran inline because they were nested.
  uint64_t nested_inline_dispatches = 0;
  /// Number of dispatches that ran inline for size or participant limits.
  uint64_t limited_inline_dispatches = 0;

  bool operator==(const CpuExecutorCounters &) const = default;
};

/**
 * Describes the estimated work performed by one independent loop iteration.
 *
 * Byte counts are exact logical traffic. ``compute_cycles`` is a relative
 * estimate of arithmetic work, excluding loads and stores.
 */
struct CpuLoopCost {
  double bytes_read = 0.0;
  double bytes_written = 0.0;
  double compute_cycles = 0.0;

  bool operator==(const CpuLoopCost &) const = default;
};

/** Cost-model decision consumed by :cpp:class:`CpuExecutor`. */
struct CpuParallelPlan {
  /// Minimum iterations needed to justify another participant.
  int64_t grain_size = 1;
  /// Maximum participants selected for this loop, including the caller.
  uint32_t participants = 1;

  bool operator==(const CpuParallelPlan &) const = default;
};

/** Kernel constraints applied to a cost-model decision. */
struct CpuParallelConstraints {
  /// Hard participant ceiling. Zero uses the session limit.
  uint32_t maximum_participants = 0;
  /// Exact participant target once parallel execution is worthwhile.
  /// Zero leaves the participant count entirely to the cost model.
  uint32_t preferred_participants = 0;

  bool operator==(const CpuParallelConstraints &) const = default;
};

/**
 * Returns the immutable sharing key for a resolved policy.
 *
 * @param policy The resolved CPU policy.
 *
 * Returns:
 *   The behavior-only executor key.
 */
CpuExecutorKey MakeCpuExecutorKey(const ResolvedCpuExecutionPolicy &policy);

/**
 * Executes range work on one caller and a persistent set of workers.
 *
 * Instances are obtained from :cpp:class:`CpuExecutorRegistry`. Concurrent
 * regions sharing an executor serialize dispatch metadata while their
 * surrounding inference calls remain independent. Nested regions execute
 * inline to avoid deadlock and oversubscription.
 */
class CpuExecutor {
public:
  CpuExecutor(const CpuExecutor &) = delete;
  CpuExecutor &operator=(const CpuExecutor &) = delete;
  ~CpuExecutor();

  /**
   * Creates an executor that delegates parallel blocks to an external worker pool.
   *
   * The executor never creates worker threads. ``maximum_participants`` limits
   * the number of blocks submitted for one region and must be positive. The
   * dispatcher must remain valid until the returned executor is destroyed.
   * Each invocation supplies its transient dispatcher context through
   * :cpp:class:`CpuExecutorDispatchScope`.
   *
   * Returns:
   *   An executor backed by ``dispatch``.
   */
  static std::unique_ptr<CpuExecutor> CreateExternal(uint32_t maximum_participants,
                                                     CpuParallelDispatchFn dispatch);

  /// Returns the effective participant count, including the caller.
  uint32_t effective_threads() const noexcept;

  /// Returns the immutable resolved policy.
  const ResolvedCpuExecutionPolicy &policy() const noexcept;

  /// Returns the immutable registry-sharing key.
  const CpuExecutorKey &key() const noexcept;

  /**
   * Returns the process-local identity of this executor instance.
   *
   * Compatible sessions that share one lease observe the same identifier.
   * The identifier is diagnostic only and must not be persisted as a tuning
   * or cache key.
   */
  uint64_t instance_id() const noexcept;

  /// Enables optional dispatch counters. Repeated calls preserve existing counts.
  void EnableCounters();

  /// Returns whether optional dispatch counters are enabled.
  bool counters_enabled() const noexcept;

  /// Returns a consistent snapshot of the optional dispatch counters.
  CpuExecutorCounters counters() const noexcept;

  /**
   * Plans a loop from its per-iteration memory and compute cost.
   *
   * The model amortizes executor startup and per-participant overhead, then
   * chooses a task grain large enough to keep dispatch overhead bounded.
   * ``maximum_participants == 0`` uses the session limit.
   */
  CpuParallelPlan PlanParallelFor(int64_t total, const CpuLoopCost &cost,
                                  uint32_t maximum_participants = 0) const noexcept;

  /** Plans a loop with explicit kernel participant constraints. */
  CpuParallelPlan PlanParallelFor(int64_t total, const CpuLoopCost &cost,
                                  const CpuParallelConstraints &constraints) const noexcept;

  /**
   * Executes contiguous ranges covering ``[0, total)``.
   *
   * ``maximum_participants == 0`` uses the session limit. A positive value may
   * lower but never raise that limit. Work below ``grain`` runs inline.
   * Executors inherited across ``fork`` are rejected.
   *
   * @param total Number of iterations. Values ``<= 0`` are a no-op.
   * @param grain Minimum iterations per parallel range. Must be positive.
   * @param context Opaque context passed to ``function``.
   * @param function Range callback, which must not throw.
   * @param maximum_participants Optional kernel-specific participant limit.
   */
  void ParallelFor(int64_t total, int64_t grain, void *context, ParallelRangeFn function,
                   uint32_t maximum_participants = 0, ParallelRegionCollector *collector = nullptr,
                   std::string_view label = {},
                   std::source_location location = std::source_location::current());

  /** Executes a loop using :cpp:func:`PlanParallelFor`. */
  void ParallelFor(int64_t total, const CpuLoopCost &cost, void *context, ParallelRangeFn function,
                   uint32_t maximum_participants = 0, ParallelRegionCollector *collector = nullptr,
                   std::string_view label = {},
                   std::source_location location = std::source_location::current());

  /** Executes a cost-aware loop with explicit kernel participant constraints. */
  void ParallelFor(int64_t total, const CpuLoopCost &cost, void *context, ParallelRangeFn function,
                   const CpuParallelConstraints &constraints,
                   ParallelRegionCollector *collector = nullptr, std::string_view label = {},
                   std::source_location location = std::source_location::current());

private:
  friend class CpuExecutorRegistry;
  struct Impl;

  explicit CpuExecutor(ResolvedCpuExecutionPolicy policy);
  CpuExecutor(ResolvedCpuExecutionPolicy policy, CpuParallelDispatchFn dispatch);

  std::unique_ptr<Impl> impl_;
};

/**
 * Returns the executor installed on the calling thread.
 *
 * A session installs its leased executor for the duration of a run so portable
 * helpers dispatch through it instead of a hidden process-wide pool.
 *
 * Returns:
 *   The installed executor, or ``nullptr`` when the thread runs outside any
 *   executor scope.
 */
CpuExecutor *CurrentCpuExecutor() noexcept;

/**
 * Installs a non-owning executor view on the calling thread.
 *
 * The previous view is restored when the scope ends, so nested scopes and
 * subgraph executions compose. The scope does not extend the executor
 * lifetime: the installer must keep its lease alive.
 */
class CpuExecutorScope {
public:
  /// Installs ``executor`` (possibly ``nullptr``) on the calling thread.
  explicit CpuExecutorScope(CpuExecutor *executor) noexcept;

  CpuExecutorScope(const CpuExecutorScope &) = delete;
  CpuExecutorScope &operator=(const CpuExecutorScope &) = delete;

  ~CpuExecutorScope();

private:
  CpuExecutor *previous_;
};

/**
 * Installs the transient context used by an external executor dispatch.
 *
 * The binding is thread-local, composes across nested scopes, and does not own
 * either pointer. It separates an invocation-specific runtime context from the
 * persistent :cpp:class:`CpuExecutor`.
 */
class CpuExecutorDispatchScope {
public:
  CpuExecutorDispatchScope(CpuExecutor *executor, void *dispatch_context) noexcept;

  CpuExecutorDispatchScope(const CpuExecutorDispatchScope &) = delete;
  CpuExecutorDispatchScope &operator=(const CpuExecutorDispatchScope &) = delete;

  ~CpuExecutorDispatchScope();

private:
  CpuExecutor *previous_executor_;
  void *previous_context_;
};

/**
 * Leases compatible shared executors from a process-owned bounded registry.
 *
 * The registry retains multithreaded executors so compatible sessions reuse
 * the same operating-system threads. Bounded spinning ends in a parked wait,
 * so an idle retained executor eventually consumes no CPU regardless of its
 * spin policy. Idle executors are evicted when the bounded capacity is needed
 * for another policy. Serial executors have no workers and are not retained.
 */
class CpuExecutorRegistry {
public:
  /// Creates a registry with a strict live-pool capacity.
  explicit CpuExecutorRegistry(size_t capacity);

  CpuExecutorRegistry(const CpuExecutorRegistry &) = delete;
  CpuExecutorRegistry &operator=(const CpuExecutorRegistry &) = delete;

  /// Resolves and acquires an executor for ``request``.
  std::shared_ptr<CpuExecutor> Acquire(const CpuExecutionPolicy &request);

  /// Acquires an executor for an already resolved policy.
  std::shared_ptr<CpuExecutor> Acquire(const ResolvedCpuExecutionPolicy &policy);

  /// Returns the configured maximum number of simultaneously live pools.
  size_t capacity() const noexcept { return capacity_; }

  /// Returns the number of live pools currently tracked.
  size_t live_pool_count();

private:
  struct Entry {
    CpuExecutorKey key;
    std::weak_ptr<CpuExecutor> executor;
    std::shared_ptr<CpuExecutor> retained_executor;
  };

  void ResetAfterForkLocked(uint64_t process_id);
  void RemoveExpiredLocked();
  bool EvictIdleRetainedLocked();

  size_t capacity_;
  uint64_t process_id_;
  std::mutex mutex_;
  std::vector<Entry> entries_;
};

/// Default maximum number of simultaneously live executor pools.
inline constexpr size_t kDefaultCpuExecutorRegistryCapacity = 8;

/**
 * Returns the process-wide executor registry.
 *
 * Returns:
 *   The bounded process-owned registry.
 */
CpuExecutorRegistry &GlobalCpuExecutorRegistry();

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
