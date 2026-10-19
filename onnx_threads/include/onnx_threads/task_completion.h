// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_threads/config.h"

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

struct TaskId {
  uint64_t value = 0;

  constexpr bool operator==(const TaskId &) const noexcept = default;
};

enum class TaskStatus {
  kPending,
  kRunning,
  kSucceeded,
  kFailed,
  kCancelled,
  kSuppressed,
};

struct TaskDiagnostic {
  TaskId task_id;
  TaskStatus status = TaskStatus::kPending;
  std::string message;
  std::optional<TaskId> caused_by;
};

/**
 * Shares completion, cancellation, and failure state for runtime operations.
 */
class ONNX_THREADS_API TaskCompletion {
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
