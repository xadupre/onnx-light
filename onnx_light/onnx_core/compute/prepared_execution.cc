// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "prepared_execution.h"

#include <functional>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

size_t PreparedKeyHash::operator()(const PreparedKey &key) const noexcept {
  return std::hash<std::string>{}(key.value);
}

RawBuffer *PreparationArena::Allocate(size_t n_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  return ExecutionArena::Allocate(n_bytes);
}

void PreparationArena::Free(RawBuffer *buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  ExecutionArena::Free(buffer);
}

size_t PreparationArena::TotalAllocatedSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ExecutionArena::TotalAllocatedSize();
}

size_t PreparationArena::PeakAllocatedSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ExecutionArena::PeakAllocatedSize();
}

void PreparationArena::ResetPeak() {
  std::lock_guard<std::mutex> lock(mutex_);
  ExecutionArena::ResetPeak();
}

RawBuffer *PreparedArena::Allocate(size_t n_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  return ExecutionArena::Allocate(n_bytes);
}

void PreparedArena::Free(RawBuffer *buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  ExecutionArena::Free(buffer);
}

size_t PreparedArena::TotalAllocatedSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ExecutionArena::TotalAllocatedSize();
}

size_t PreparedArena::PeakAllocatedSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ExecutionArena::PeakAllocatedSize();
}

void PreparedArena::ResetPeak() {
  std::lock_guard<std::mutex> lock(mutex_);
  ExecutionArena::ResetPeak();
}

struct PreparedObjectStore::Entry {
  explicit Entry(PreparedObjectRequirement requirement_)
      : requirement(std::move(requirement_)),
        completion(std::make_shared<TaskCompletion>(TaskId{})),
        allocation(std::make_shared<AllocationHandle>()) {}

  PreparedObjectRequirement requirement;
  PreparedResidencyState state = PreparedResidencyState::kAbsent;
  uint64_t generation = 0;
  std::shared_ptr<TaskCompletion> completion;
  std::shared_ptr<AllocationHandle> allocation;
};

PreparedObjectStore::PreparedObjectStore() = default;

PreparedObjectStore::~PreparedObjectStore() = default;

void PreparedObjectStore::ValidateGeneration(const Entry &entry,
                                             const PreparedObjectRequest &request) {
  EXT_ENFORCE(entry.generation == request.generation,
              "Prepared object request refers to a stale generation.");
}

PreparedObjectRequest PreparedObjectStore::Request(const PreparedObjectRequirement &requirement) {
  EXT_ENFORCE(!requirement.key.value.empty(), "Prepared object keys must not be empty.");
  std::lock_guard<std::mutex> lock(mutex_);
  auto [found, inserted] =
      entries_.try_emplace(requirement.key, std::make_unique<Entry>(requirement));
  Entry &entry = *found->second;
  if (!inserted) {
    EXT_ENFORCE(entry.requirement.source_fallback == requirement.source_fallback,
                "Prepared object key '", requirement.key.value,
                "' was requested with a different source fallback.");
  }

  const bool produce = entry.state == PreparedResidencyState::kAbsent ||
                       entry.state == PreparedResidencyState::kFailed ||
                       entry.state == PreparedResidencyState::kPersisted;
  if (produce) {
    auto completion = std::make_shared<TaskCompletion>(TaskId{entry.generation + 1});
    ++entry.generation;
    entry.state = PreparedResidencyState::kLoading;
    entry.completion = std::move(completion);
  }
  return PreparedObjectRequest{entry.requirement.key, entry.generation, *entry.completion, produce};
}

void PreparedObjectStore::MarkPreparing(const PreparedObjectRequest &request) {
  std::lock_guard<std::mutex> lock(mutex_);
  Entry &entry = *entries_.at(request.key);
  ValidateGeneration(entry, request);
  EXT_ENFORCE(request.producer && entry.state == PreparedResidencyState::kLoading,
              "Only the loading producer can start preparing an object.");
  entry.state = PreparedResidencyState::kPreparing;
}

void PreparedObjectStore::Publish(const PreparedObjectRequest &request,
                                  AllocationHandle allocation) {
  EXT_ENFORCE(allocation, "A prepared publication must own a complete allocation.");
  std::lock_guard<std::mutex> lock(mutex_);
  Entry &entry = *entries_.at(request.key);
  ValidateGeneration(entry, request);
  EXT_ENFORCE(request.producer && (entry.state == PreparedResidencyState::kLoading ||
                                   entry.state == PreparedResidencyState::kPreparing),
              "Only the current producer can publish a prepared object.");
  *entry.allocation = std::move(allocation);
  entry.state = PreparedResidencyState::kResident;
  ++readiness_epoch_;
  entry.completion->Succeed();
}

void PreparedObjectStore::Fail(const PreparedObjectRequest &request, std::exception_ptr error,
                               std::string message) {
  std::lock_guard<std::mutex> lock(mutex_);
  Entry &entry = *entries_.at(request.key);
  ValidateGeneration(entry, request);
  EXT_ENFORCE(request.producer && (entry.state == PreparedResidencyState::kLoading ||
                                   entry.state == PreparedResidencyState::kPreparing),
              "Only the current producer can fail a prepared object.");
  entry.state = PreparedResidencyState::kFailed;
  entry.completion->Fail(std::move(error), std::move(message));
}

std::optional<PreparedObjectView> PreparedObjectStore::Find(const PreparedKey &key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = entries_.find(key);
  if (found == entries_.end() || found->second->state != PreparedResidencyState::kResident) {
    return std::nullopt;
  }
  const Entry &entry = *found->second;
  return PreparedObjectView{entry.allocation->buffer(), entry.allocation->owner(), entry.generation,
                            entry.allocation};
}

bool PreparedObjectStore::Evict(const PreparedKey &key) {
  std::shared_ptr<AllocationHandle> allocation;
  auto replacement = std::make_shared<AllocationHandle>();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(key);
    if (found == entries_.end() || found->second->state != PreparedResidencyState::kResident) {
      return false;
    }
    Entry &entry = *found->second;
    entry.state = PreparedResidencyState::kEvicting;
    allocation = std::move(entry.allocation);
    entry.allocation = std::move(replacement);
    entry.state = PreparedResidencyState::kAbsent;
    ++readiness_epoch_;
  }
  allocation.reset();
  return true;
}

PreparedResidencyState PreparedObjectStore::State(const PreparedKey &key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = entries_.find(key);
  return found == entries_.end() ? PreparedResidencyState::kAbsent : found->second->state;
}

MaterializationTaskDescriptors
ExpandMaterializationRecipe(const PreparedRequirementDescriptor &requirement,
                            const MaterializationRecipe &selected, TaskId first_task_id) {
  EXT_ENFORCE(!selected.payload_id.empty(), "A materialization recipe must name its payload.");
  const TaskId load_id{first_task_id.value};
  const TaskId prepack_id{first_task_id.value + 1};
  const TaskId publish_id{first_task_id.value + 2};
  const TaskId fallback_id{first_task_id.value + 3};

  MaterializationTaskDescriptors descriptors;
  descriptors.load =
      TaskDescriptor{load_id, TaskScope::kSession, TaskKind::kReadPayload, ResourceClass::kIo};
  descriptors.prepack = TaskDescriptor{
      prepack_id, TaskScope::kSession, TaskKind::kPrepare, ResourceClass::kCpu, {load_id}};
  descriptors.publish = TaskDescriptor{
      publish_id, TaskScope::kSession, TaskKind::kPublish, ResourceClass::kInline, {prepack_id}};
  descriptors.publish.publishes = requirement.requirement.key;
  descriptors.dormant_fallback =
      TaskDescriptor{fallback_id, TaskScope::kSession, TaskKind::kFallback, ResourceClass::kIo};
  descriptors.dormant_fallback.publishes = requirement.requirement.key;
  descriptors.dormant_fallback.dormant = true;
  return descriptors;
}

PreparedExecutionPlan::PreparedExecutionPlan(std::vector<TaskDescriptor> tasks)
    : tasks_(std::move(tasks)) {
  std::unordered_map<uint64_t, TaskScope> scopes;
  std::unordered_set<uint64_t> dormant;
  for (const TaskDescriptor &task : tasks_) {
    EXT_ENFORCE(task.id.value != 0, "Prepared task IDs must not be zero.");
    EXT_ENFORCE(scopes.emplace(task.id.value, task.scope).second, "Duplicate prepared task ID ",
                task.id.value, ".");
    for (const TaskId dependency : task.dependencies) {
      const auto found = scopes.find(dependency.value);
      EXT_ENFORCE(found != scopes.end(), "Prepared task ", task.id.value,
                  " depends on missing or later task ", dependency.value, ".");
      EXT_ENFORCE(task.scope != TaskScope::kSession || found->second == TaskScope::kSession,
                  "Session task ", task.id.value, " cannot depend on invocation task ",
                  dependency.value, ".");
      EXT_ENFORCE(dormant.count(dependency.value) == 0, "Prepared task ", task.id.value,
                  " cannot depend on dormant task ", dependency.value, ".");
    }
    if (task.dormant) {
      dormant.insert(task.id.value);
    }
  }
}

namespace {

bool DependencySucceeded(const TaskCompletion &dependency, TaskCompletion &completion,
                         TaskId dependency_id) {
  try {
    dependency.Wait();
  } catch (...) {
    completion.Suppress(dependency_id, "Prepared task dependency failed.");
    return false;
  }
  if (dependency.status() != TaskStatus::kSucceeded) {
    completion.Suppress(dependency_id, "Prepared task dependency did not succeed.");
    return false;
  }
  return true;
}

} // namespace

PreparedExecutionResult
PreparedExecutionPlan::RunSequential(PreparedExecutionState &state,
                                     const PreparedTaskExecutor &executor) const {
  EXT_ENFORCE(static_cast<bool>(executor), "Prepared execution requires a task executor.");
  const uint64_t invocation_id = ++state.next_invocation_id_;
  std::unordered_map<uint64_t, PreparedExecutionState::SessionTaskRequest> session;
  std::unordered_map<uint64_t, TaskCompletion> invocation;

  for (const TaskDescriptor &task : tasks_) {
    if (task.dormant) {
      continue;
    }
    if (task.scope == TaskScope::kSession) {
      session.emplace(task.id.value, state.RequestSessionTask(task.id));
    } else {
      invocation.emplace(task.id.value, TaskCompletion(task.id));
    }
  }

  bool has_producers = false;
  for (const TaskDescriptor &task : tasks_) {
    if (!task.dormant && task.scope == TaskScope::kSession && session.at(task.id.value).producer) {
      has_producers = true;
      break;
    }
  }
  std::thread producer;
  if (has_producers) {
    producer = std::thread([&]() {
      for (const TaskDescriptor &task : tasks_) {
        if (task.dormant || task.scope != TaskScope::kSession) {
          continue;
        }
        auto request = session.at(task.id.value);
        if (!request.producer) {
          continue;
        }
        request.completion.MarkRunning();
        bool ready = true;
        for (const TaskId dependency_id : task.dependencies) {
          if (!DependencySucceeded(session.at(dependency_id.value).completion, request.completion,
                                   dependency_id)) {
            ready = false;
            break;
          }
        }
        if (!ready) {
          continue;
        }
        try {
          executor(task, state);
          request.completion.Succeed();
        } catch (...) {
          request.completion.Fail(std::current_exception(), "Prepared session task failed.");
        }
      }
    });
  }

  for (const TaskDescriptor &task : tasks_) {
    if (task.dormant || task.scope != TaskScope::kInvocation) {
      continue;
    }
    TaskCompletion &completion = invocation.at(task.id.value);
    bool ready = true;
    for (const TaskId dependency_id : task.dependencies) {
      const auto session_dependency = session.find(dependency_id.value);
      const TaskCompletion &dependency = session_dependency != session.end()
                                             ? session_dependency->second.completion
                                             : invocation.at(dependency_id.value);
      if (!DependencySucceeded(dependency, completion, dependency_id)) {
        ready = false;
        break;
      }
    }
    if (!ready) {
      continue;
    }
    completion.MarkRunning();
    try {
      executor(task, state);
      completion.Succeed();
    } catch (...) {
      completion.Fail(std::current_exception(), "Prepared invocation task failed.");
    }
  }
  if (producer.joinable()) {
    producer.join();
  }

  PreparedExecutionResult result;
  result.invocation_id = invocation_id;
  result.diagnostics.reserve(tasks_.size());
  result.session_generations.reserve(session.size());
  for (const TaskDescriptor &task : tasks_) {
    if (task.dormant) {
      result.diagnostics.push_back(
          TaskDiagnostic{task.id, TaskStatus::kPending, std::string(), std::nullopt});
    } else if (task.scope == TaskScope::kSession) {
      result.diagnostics.push_back(session.at(task.id.value).completion.diagnostic());
      result.session_generations.emplace_back(task.id, session.at(task.id.value).generation);
    } else {
      result.diagnostics.push_back(invocation.at(task.id.value).diagnostic());
    }
  }
  return result;
}

struct PreparedExecutionState::SessionTaskEntry {
  explicit SessionTaskEntry(TaskId task_id)
      : completion(std::make_shared<TaskCompletion>(task_id)) {}

  uint64_t generation = 1;
  std::shared_ptr<TaskCompletion> completion;
};

PreparedExecutionState::PreparedExecutionState(size_t preparation_slots, size_t prepared_slots,
                                               size_t preparation_retention_cap,
                                               size_t prepared_retention_cap)
    : preparation_arena_(preparation_slots, preparation_retention_cap),
      prepared_arena_(prepared_slots, prepared_retention_cap) {}

PreparedExecutionState::~PreparedExecutionState() = default;

PreparedExecutionState::SessionTaskRequest
PreparedExecutionState::RequestSessionTask(TaskId task_id) {
  std::lock_guard<std::mutex> lock(session_tasks_mutex_);
  auto [found, inserted] =
      session_tasks_.try_emplace(task_id.value, std::make_unique<SessionTaskEntry>(task_id));
  SessionTaskEntry &entry = *found->second;
  const TaskStatus status = entry.completion->status();
  const bool producer = inserted || status == TaskStatus::kFailed ||
                        status == TaskStatus::kCancelled || status == TaskStatus::kSuppressed;
  if (!inserted && producer) {
    ++entry.generation;
    entry.completion = std::make_shared<TaskCompletion>(task_id);
  }
  return SessionTaskRequest{entry.generation, *entry.completion, producer};
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
