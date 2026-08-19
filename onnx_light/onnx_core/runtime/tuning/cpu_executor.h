// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/tuning/cpu_execution_policy.h"
#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

/**
 * @file cpu_executor.h
 * @brief Shared CPU executor and compatible-pool registry.
 */

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/// Type-erased range callable: ``function(context, begin, end)``.
using ParallelRangeFn = void (*)(void *, int64_t, int64_t);

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

  /// Returns the effective participant count, including the caller.
  uint32_t effective_threads() const noexcept;

  /// Returns the immutable resolved policy.
  const ResolvedCpuExecutionPolicy &policy() const noexcept;

  /// Returns the immutable registry-sharing key.
  const CpuExecutorKey &key() const noexcept;

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
                   uint32_t maximum_participants = 0);

private:
  friend class CpuExecutorRegistry;
  struct Impl;

  explicit CpuExecutor(ResolvedCpuExecutionPolicy policy);

  std::unique_ptr<Impl> impl_;
};

/**
 * Leases compatible shared executors from a process-owned bounded registry.
 *
 * The registry stores weak references. Compatible callers receive the same
 * executor while at least one lease remains. Releasing the final lease stops
 * and destroys its workers immediately. Capacity limits simultaneously live
 * incompatible pools; expired entries never consume capacity.
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
  };

  void ResetAfterForkLocked(uint64_t process_id);
  void RemoveExpiredLocked();

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
