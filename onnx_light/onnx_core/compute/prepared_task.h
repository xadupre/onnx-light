// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

struct TaskId {
  uint64_t value = 0;

  constexpr bool operator==(const TaskId &) const noexcept = default;
};

struct PreparedKey {
  std::string value;

  bool operator==(const PreparedKey &) const noexcept = default;
};

enum class TaskScope {
  kSession,
  kInvocation,
};

enum class TaskKind {
  kReadPayload,
  kPrepare,
  kExecute,
  kPersist,
  kPublish,
  kFallback,
};

enum class ResourceClass {
  kInline,
  kIo,
  kCpu,
  kDevice,
};

enum class TaskPriority {
  kBackground,
  kPrefetch,
  kCritical,
};

enum class TaskStatus {
  kPending,
  kRunning,
  kSucceeded,
  kFailed,
  kCancelled,
  kSuppressed,
};

struct ActionRange {
  size_t begin = 0;
  size_t end = 0;
};

struct TaskDiagnostic {
  TaskId task_id;
  TaskStatus status = TaskStatus::kPending;
  std::string message;
  std::optional<TaskId> caused_by;
};

struct TaskDescriptor {
  TaskDescriptor() = default;
  TaskDescriptor(TaskId id_, TaskScope scope_, TaskKind kind_, ResourceClass resource_,
                 std::vector<TaskId> dependencies_ = {}, size_t estimated_input_bytes_ = 0,
                 size_t estimated_output_bytes_ = 0, size_t peak_temporary_bytes_ = 0,
                 std::optional<ActionRange> actions_ = {},
                 std::vector<PreparedKey> prepared_requirements_ = {},
                 std::optional<PreparedKey> publishes_ = {}, bool dormant_ = false,
                 std::optional<TaskPriority> priority_ = {}, std::string payload_id_ = {})
      : id(id_), scope(scope_), kind(kind_), resource(resource_),
        dependencies(std::move(dependencies_)), estimated_input_bytes(estimated_input_bytes_),
        estimated_output_bytes(estimated_output_bytes_),
        peak_temporary_bytes(peak_temporary_bytes_), actions(actions_),
        prepared_requirements(std::move(prepared_requirements_)), publishes(std::move(publishes_)),
        dormant(dormant_),
        priority(priority_.value_or(scope_ == TaskScope::kInvocation ? TaskPriority::kCritical
                                                                     : TaskPriority::kPrefetch)),
        payload_id(std::move(payload_id_)) {}

  TaskId id;
  TaskScope scope = TaskScope::kInvocation;
  TaskKind kind = TaskKind::kExecute;
  ResourceClass resource = ResourceClass::kCpu;
  std::vector<TaskId> dependencies;
  size_t estimated_input_bytes = 0;
  size_t estimated_output_bytes = 0;
  size_t peak_temporary_bytes = 0;
  std::optional<ActionRange> actions;
  std::vector<PreparedKey> prepared_requirements;
  std::optional<PreparedKey> publishes;
  bool dormant = false;
  TaskPriority priority = TaskPriority::kPrefetch;
  std::string payload_id;
};

/**
 * Shares the completion state of one task between producers and dependants.
 */
class ONNX_LIGHT_CORE_API TaskCompletion {
public:
  explicit TaskCompletion(TaskId task_id);

  /** Marks the task as running. */
  void MarkRunning() const;

  /** Marks the task as successfully completed. */
  void Succeed() const;

  /** Marks the task as failed and preserves its exception for Wait(). */
  void Fail(std::exception_ptr error, std::string message = {}) const;

  /** Marks a pending task as cancelled. */
  void Cancel(std::string message = {}) const;

  /** Marks a pending task as suppressed by a failed dependency. */
  void Suppress(TaskId caused_by, std::string message = {}) const;

  /** Waits for a terminal state and rethrows a task exception. */
  void Wait() const;

  /** Returns whether the task has reached a terminal state. */
  bool IsReady() const;

  TaskStatus status() const;
  TaskDiagnostic diagnostic() const;

private:
  struct State {
    explicit State(TaskId id) { diagnostic.task_id = id; }

    mutable std::mutex mutex;
    std::condition_variable completed;
    TaskDiagnostic diagnostic;
    std::exception_ptr error;
  };

  static bool IsTerminal(TaskStatus status);
  std::shared_ptr<State> state_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
