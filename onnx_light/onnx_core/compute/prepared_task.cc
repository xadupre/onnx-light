// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "prepared_task.h"

#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

TaskCompletion::TaskCompletion(TaskId task_id) : state_(std::make_shared<State>(task_id)) {}

bool TaskCompletion::IsTerminal(TaskStatus status) {
  return status == TaskStatus::kSucceeded || status == TaskStatus::kFailed ||
         status == TaskStatus::kCancelled || status == TaskStatus::kSuppressed;
}

void TaskCompletion::MarkRunning() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  EXT_ENFORCE(state_->diagnostic.status == TaskStatus::kPending, "Only a pending task can start.");
  state_->diagnostic.status = TaskStatus::kRunning;
}

void TaskCompletion::Succeed() const {
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    EXT_ENFORCE(!IsTerminal(state_->diagnostic.status), "Task is already complete.");
    state_->diagnostic.status = TaskStatus::kSucceeded;
  }
  state_->completed.notify_all();
}

void TaskCompletion::Fail(std::exception_ptr error, std::string message) const {
  EXT_ENFORCE(error != nullptr, "A failed task must provide an exception.");
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    EXT_ENFORCE(!IsTerminal(state_->diagnostic.status), "Task is already complete.");
    state_->diagnostic.status = TaskStatus::kFailed;
    state_->diagnostic.message = std::move(message);
    state_->error = std::move(error);
  }
  state_->completed.notify_all();
}

void TaskCompletion::Cancel(std::string message) const {
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    EXT_ENFORCE(state_->diagnostic.status == TaskStatus::kPending,
                "Only a pending task can be cancelled.");
    state_->diagnostic.status = TaskStatus::kCancelled;
    state_->diagnostic.message = std::move(message);
  }
  state_->completed.notify_all();
}

void TaskCompletion::Suppress(TaskId caused_by, std::string message) const {
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    EXT_ENFORCE(state_->diagnostic.status == TaskStatus::kPending,
                "Only a pending task can be suppressed.");
    state_->diagnostic.status = TaskStatus::kSuppressed;
    state_->diagnostic.message = std::move(message);
    state_->diagnostic.caused_by = caused_by;
  }
  state_->completed.notify_all();
}

void TaskCompletion::Wait() const {
  std::exception_ptr error;
  {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->completed.wait(lock, [this]() { return IsTerminal(state_->diagnostic.status); });
    error = state_->error;
  }
  if (error != nullptr)
    std::rethrow_exception(error);
}

bool TaskCompletion::IsReady() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return IsTerminal(state_->diagnostic.status);
}

TaskStatus TaskCompletion::status() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->diagnostic.status;
}

TaskDiagnostic TaskCompletion::diagnostic() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->diagnostic;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
