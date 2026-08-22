// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/prepared_task.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

struct PreparedKeyHash {
  size_t operator()(const PreparedKey &key) const noexcept;
};

enum class PreparedResidencyState {
  kAbsent,
  kLoading,
  kPreparing,
  kResident,
  kFailed,
  kEvicting,
  kPersisted,
};

struct PreparedObjectRequirement {
  PreparedKey key;
  std::string source_fallback;
};

struct PreparedObjectRequest {
  PreparedKey key;
  uint64_t generation = 0;
  TaskCompletion completion;
  bool producer = false;
};

struct PreparedObjectView {
  const RawBuffer *buffer = nullptr;
  RawBufferAllocator *owner = nullptr;
  uint64_t generation = 0;
  std::shared_ptr<const AllocationHandle> pin;
};

class ONNX_LIGHT_CORE_API PreparationArena final : public ExecutionArena {
public:
  using ExecutionArena::ExecutionArena;

  RawBuffer *Allocate(size_t n_bytes) override;
  void Free(RawBuffer *buffer) override;
  size_t TotalAllocatedSize() const override;
  size_t PeakAllocatedSize() const override;
  void ResetPeak() override;

private:
  mutable std::mutex mutex_;
};

class ONNX_LIGHT_CORE_API PreparedArena final : public ExecutionArena {
public:
  using ExecutionArena::ExecutionArena;

  RawBuffer *Allocate(size_t n_bytes) override;
  void Free(RawBuffer *buffer) override;
  size_t TotalAllocatedSize() const override;
  size_t PeakAllocatedSize() const override;
  void ResetPeak() override;

private:
  mutable std::mutex mutex_;
};

/**
 * Tracks immutable prepared objects and shares one in-flight generation per key.
 */
class ONNX_LIGHT_CORE_API PreparedObjectStore {
public:
  PreparedObjectStore();
  ~PreparedObjectStore();

  /** Returns the resident generation or elects one caller to produce a new one. */
  PreparedObjectRequest Request(const PreparedObjectRequirement &requirement);

  /** Transitions the request selected as producer from loading to preparing. */
  void MarkPreparing(const PreparedObjectRequest &request);

  /**
   * Atomically installs a complete allocation before satisfying its completion.
   */
  void Publish(const PreparedObjectRequest &request, AllocationHandle allocation);

  /** Fails the current generation and preserves its diagnostic exception. */
  void Fail(const PreparedObjectRequest &request, std::exception_ptr error,
            std::string message = {});

  /** Returns a resident object's immutable view, or no value when not resident. */
  std::optional<PreparedObjectView> Find(const PreparedKey &key) const;

  /** Evicts a resident generation and returns its allocation to its owner. */
  bool Evict(const PreparedKey &key);

  PreparedResidencyState State(const PreparedKey &key) const;
  uint64_t readiness_epoch() const noexcept { return readiness_epoch_.load(); }

private:
  struct Entry;

  static void ValidateGeneration(const Entry &entry, const PreparedObjectRequest &request);

  mutable std::mutex mutex_;
  std::unordered_map<PreparedKey, std::unique_ptr<Entry>, PreparedKeyHash> entries_;
  std::atomic<uint64_t> readiness_epoch_{0};
};

enum class MaterializationRecipeKind {
  kReadPackedPayload,
  kReadSourceAndPrepack,
};

struct MaterializationRecipe {
  MaterializationRecipeKind kind = MaterializationRecipeKind::kReadSourceAndPrepack;
  std::string payload_id;
  std::string layout;
};

struct PreparedRequirementDescriptor {
  PreparedObjectRequirement requirement;
  std::vector<MaterializationRecipe> recipes;
};

struct MaterializationTaskDescriptors {
  TaskDescriptor load;
  TaskDescriptor prepack;
  TaskDescriptor publish;
  TaskDescriptor dormant_fallback;
};

class PreparedExecutionState;

using PreparedTaskExecutor = std::function<void(const TaskDescriptor &, PreparedExecutionState &)>;

struct PreparedExecutionResult {
  uint64_t invocation_id = 0;
  std::vector<TaskDiagnostic> diagnostics;
  std::vector<std::pair<TaskId, uint64_t>> session_generations;
  bool used_hot_path = false;
  size_t enqueued_tasks = 0;
  size_t continuation_suspensions = 0;
  size_t peak_in_flight_bytes = 0;
};

struct PreparedSchedulerOptions {
  size_t io_workers = 2;
  size_t global_memory_budget = std::numeric_limits<size_t>::max();
  size_t preparation_memory_budget = std::numeric_limits<size_t>::max();
  size_t prepared_memory_budget = std::numeric_limits<size_t>::max();
  size_t execution_memory_budget = std::numeric_limits<size_t>::max();
  size_t io_memory_budget = std::numeric_limits<size_t>::max();
  size_t reserved_critical_memory = 0;
};

/**
 * Owns the completion state of one ``PrepareAsync``/``RunAsync`` submission.
 *
 * The handle is movable but not copyable. It keeps the submission's borrowed
 * ``PreparedExecutionState`` and task executor referenced by the caller alive
 * for as long as the submission still runs: destroying (or move-assigning
 * over) a live handle blocks until every task the submission still owns
 * reaches a terminal state, so no detached task can outlive the context or
 * input state its submission borrowed. ``Cancel()`` only affects tasks this
 * submission produces; shared session-scoped work still required elsewhere
 * is left untouched.
 */
class ONNX_LIGHT_CORE_API ExecutionHandle final {
public:
  ExecutionHandle();
  ExecutionHandle(ExecutionHandle &&) noexcept;
  ExecutionHandle &operator=(ExecutionHandle &&) noexcept;
  ExecutionHandle(const ExecutionHandle &) = delete;
  ExecutionHandle &operator=(const ExecutionHandle &) = delete;
  ~ExecutionHandle();

  /** Blocks until every required task finishes and rethrows the first error. */
  const PreparedExecutionResult &Wait();

  /** Returns whether the submission has already reached a terminal state. */
  bool IsReady() const;

  /**
   * Cooperatively cancels tasks this submission has not yet dispatched.
   *
   * Already-running tasks finish normally; downstream tasks are suppressed.
   * Session-scoped work owned by another submission is left untouched.
   */
  void Cancel();

private:
  friend class PreparedExecutionPlan;
  struct Shared;

  static ExecutionHandle Submit(std::function<PreparedExecutionResult(std::atomic<bool> &)> work);

  std::shared_ptr<Shared> shared_;
};

/**
 * Describes session preparation and invocation execution in one immutable graph.
 */
class ONNX_LIGHT_CORE_API PreparedExecutionPlan {
public:
  explicit PreparedExecutionPlan(std::vector<TaskDescriptor> tasks);

  const std::vector<TaskDescriptor> &tasks() const noexcept { return tasks_; }

  /**
   * Executes session producers and the invocation tasks through completion events.
   *
   * @param state Mutable session state that owns reusable session-task generations.
   * @param executor Executes a task descriptor. It may be called concurrently by
   *                 the session worker and the invoking thread.
   * @return Fresh invocation diagnostics and the session generations it observed.
   */
  PreparedExecutionResult RunSequential(PreparedExecutionState &state,
                                        const PreparedTaskExecutor &executor) const;

  /**
   * Executes ready tasks using persistent bounded I/O workers and ``cpu_executor``.
   *
   * CPU work is dispatched only through the supplied session executor; this
   * method never creates another CPU worker pool.
   */
  PreparedExecutionResult RunParallel(PreparedExecutionState &state,
                                      const PreparedTaskExecutor &executor,
                                      CpuExecutor &cpu_executor) const;

  /**
   * Submits only this plan's session-scoped tasks and returns immediately.
   *
   * ``state``, ``executor``, and (when supplied) ``cpu_executor`` must outlive
   * the returned handle.
   */
  ExecutionHandle PrepareAsync(PreparedExecutionState &state, const PreparedTaskExecutor &executor,
                              CpuExecutor *cpu_executor = nullptr) const;

  /**
   * Submits every task in this plan and returns immediately.
   *
   * ``state``, ``executor``, and (when supplied) ``cpu_executor`` must outlive
   * the returned handle.
   */
  ExecutionHandle RunAsync(PreparedExecutionState &state, const PreparedTaskExecutor &executor,
                          CpuExecutor *cpu_executor = nullptr) const;

  /** Synchronous ``PrepareAsync`` followed by ``Wait``. */
  void Prepare(PreparedExecutionState &state, const PreparedTaskExecutor &executor,
              CpuExecutor *cpu_executor = nullptr) const;

private:
  PreparedExecutionResult Run(PreparedExecutionState &state, const PreparedTaskExecutor &executor,
                              CpuExecutor *cpu_executor,
                              const std::atomic<bool> *cancel_requested = nullptr) const;

  std::vector<TaskDescriptor> tasks_;
  std::vector<PreparedKey> prepared_requirements_;
};

/**
 * Expands one selected recipe into its session-scoped preparation task chain.
 */
ONNX_LIGHT_CORE_API MaterializationTaskDescriptors
ExpandMaterializationRecipe(const PreparedRequirementDescriptor &requirement,
                            const MaterializationRecipe &selected, TaskId first_task_id);

/**
 * Owns session preparation arenas and the mutable prepared-object residency state.
 */
class ONNX_LIGHT_CORE_API PreparedExecutionState {
public:
  explicit PreparedExecutionState(
      size_t preparation_slots = 4, size_t prepared_slots = 4,
      size_t preparation_retention_cap = std::numeric_limits<size_t>::max(),
      size_t prepared_retention_cap = std::numeric_limits<size_t>::max(),
      PreparedSchedulerOptions scheduler_options = {});
  ~PreparedExecutionState();

  PreparationArena &preparation_arena() noexcept { return preparation_arena_; }
  PreparedArena &prepared_arena() noexcept { return prepared_arena_; }
  PreparedObjectStore &objects() noexcept { return objects_; }
  const PreparedObjectStore &objects() const noexcept { return objects_; }
  uint64_t readiness_epoch() const noexcept { return objects_.readiness_epoch(); }

private:
  friend class PreparedExecutionPlan;
  struct SessionTaskEntry;
  struct SchedulerState;
  struct SessionTaskRequest {
    uint64_t generation = 0;
    TaskCompletion completion{TaskId{}};
    bool producer = false;
  };

  SessionTaskRequest RequestSessionTask(TaskId task_id, bool force_retry);

  PreparationArena preparation_arena_;
  PreparedArena prepared_arena_;
  PreparedObjectStore objects_;
  std::mutex session_tasks_mutex_;
  std::unordered_map<uint64_t, std::unique_ptr<SessionTaskEntry>> session_tasks_;
  std::atomic<uint64_t> next_invocation_id_{0};
  std::unique_ptr<SchedulerState> scheduler_;
  std::mutex hot_path_mutex_;
  const PreparedExecutionPlan *hot_path_plan_ = nullptr;
  uint64_t hot_path_epoch_ = 0;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
