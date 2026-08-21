// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "prepared_execution.h"

#include "worker_pool.h"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <limits>
#include <stdexcept>
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
  std::unordered_set<std::string> prepared_requirements;
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
    for (const PreparedKey &key : task.prepared_requirements) {
      if (prepared_requirements.insert(key.value).second) {
        prepared_requirements_.push_back(key);
      }
    }
  }
}

namespace {

int PriorityRank(TaskPriority priority) { return static_cast<int>(priority); }

size_t EstimatedBytes(const TaskDescriptor &task) {
  const size_t first =
      task.estimated_input_bytes > std::numeric_limits<size_t>::max() - task.estimated_output_bytes
          ? std::numeric_limits<size_t>::max()
          : task.estimated_input_bytes + task.estimated_output_bytes;
  return first > std::numeric_limits<size_t>::max() - task.peak_temporary_bytes
             ? std::numeric_limits<size_t>::max()
             : first + task.peak_temporary_bytes;
}

enum class BudgetDomain {
  kGeneral,
  kIo,
  kPreparation,
  kPrepared,
};

BudgetDomain DomainFor(const TaskDescriptor &task) {
  if (task.resource == ResourceClass::kIo) {
    return BudgetDomain::kIo;
  }
  if (task.kind == TaskKind::kPrepare) {
    return BudgetDomain::kPreparation;
  }
  if (task.kind == TaskKind::kPublish) {
    return BudgetDomain::kPrepared;
  }
  return BudgetDomain::kGeneral;
}

size_t DomainCap(const PreparedSchedulerOptions &options, BudgetDomain domain) {
  switch (domain) {
  case BudgetDomain::kIo:
    return options.io_memory_budget;
  case BudgetDomain::kPreparation:
    return options.preparation_memory_budget;
  case BudgetDomain::kPrepared:
    return options.prepared_memory_budget;
  case BudgetDomain::kGeneral:
    return std::numeric_limits<size_t>::max();
  }
  return std::numeric_limits<size_t>::max();
}

bool IsTerminal(TaskStatus status) {
  return status == TaskStatus::kSucceeded || status == TaskStatus::kFailed ||
         status == TaskStatus::kCancelled || status == TaskStatus::kSuppressed;
}

bool AllPreparedRequirementsResident(const std::vector<TaskDescriptor> &tasks,
                                     const PreparedObjectStore &objects) {
  for (const TaskDescriptor &task : tasks) {
    for (const PreparedKey &key : task.prepared_requirements) {
      if (!objects.Find(key).has_value()) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

struct PreparedExecutionState::SchedulerState {
  explicit SchedulerState(PreparedSchedulerOptions options_) : options(std::move(options_)) {
    EXT_ENFORCE(options.io_workers > 0, "Prepared scheduler requires at least one I/O worker.");
    EXT_ENFORCE(options.reserved_critical_memory <= options.global_memory_budget,
                "Reserved critical memory exceeds the global memory budget.");
    io_pool.Start(static_cast<int32_t>(options.io_workers));
  }

  bool TryAcquire(const TaskDescriptor &task, bool critical_pending, size_t &new_total) {
    const size_t bytes = EstimatedBytes(task);
    const BudgetDomain domain = DomainFor(task);
    const size_t domain_cap = DomainCap(options, domain);
    EXT_ENFORCE(bytes <= options.global_memory_budget, "Prepared task ", task.id.value,
                " exceeds the global memory budget.");
    EXT_ENFORCE(bytes <= domain_cap, "Prepared task ", task.id.value,
                " exceeds its arena memory budget.");

    std::lock_guard<std::mutex> lock(mutex);
    const bool speculative = task.priority != TaskPriority::kCritical && critical_pending;
    if (task.resource == ResourceClass::kIo) {
      const size_t io_limit = speculative ? options.io_workers - 1 : options.io_workers;
      if (active_io >= io_limit) {
        return false;
      }
    }
    const size_t global_cap = speculative
                                  ? options.global_memory_budget - options.reserved_critical_memory
                                  : options.global_memory_budget;
    size_t &domain_used = Used(domain);
    if (bytes > global_cap - std::min(global_cap, global_used) ||
        bytes > domain_cap - std::min(domain_cap, domain_used)) {
      return false;
    }
    global_used += bytes;
    domain_used += bytes;
    active_io += task.resource == ResourceClass::kIo ? 1 : 0;
    peak_global_used = std::max(peak_global_used, global_used);
    new_total = global_used;
    return true;
  }

  void Release(const TaskDescriptor &task) {
    const size_t bytes = EstimatedBytes(task);
    std::lock_guard<std::mutex> lock(mutex);
    global_used -= bytes;
    Used(DomainFor(task)) -= bytes;
    active_io -= task.resource == ResourceClass::kIo ? 1 : 0;
  }

  size_t &Used(BudgetDomain domain) {
    switch (domain) {
    case BudgetDomain::kIo:
      return io_used;
    case BudgetDomain::kPreparation:
      return preparation_used;
    case BudgetDomain::kPrepared:
      return prepared_used;
    case BudgetDomain::kGeneral:
      return general_used;
    }
    return general_used;
  }

  PreparedSchedulerOptions options;
  utils::WorkerPool io_pool;
  std::mutex mutex;
  size_t global_used = 0;
  size_t general_used = 0;
  size_t io_used = 0;
  size_t preparation_used = 0;
  size_t prepared_used = 0;
  size_t active_io = 0;
  size_t peak_global_used = 0;
};

PreparedExecutionResult
PreparedExecutionPlan::RunSequential(PreparedExecutionState &state,
                                     const PreparedTaskExecutor &executor) const {
  return Run(state, executor, nullptr);
}

PreparedExecutionResult PreparedExecutionPlan::RunParallel(PreparedExecutionState &state,
                                                           const PreparedTaskExecutor &executor,
                                                           CpuExecutor &cpu_executor) const {
  return Run(state, executor, &cpu_executor);
}

PreparedExecutionResult PreparedExecutionPlan::Run(PreparedExecutionState &state,
                                                   const PreparedTaskExecutor &executor,
                                                   CpuExecutor *cpu_executor) const {
  EXT_ENFORCE(static_cast<bool>(executor), "Prepared execution requires a task executor.");
  const uint64_t invocation_id = ++state.next_invocation_id_;
  std::unordered_map<uint64_t, PreparedExecutionState::SessionTaskRequest> session;
  std::unordered_set<uint64_t> replay_session_tasks;
  for (auto task = tasks_.rbegin(); task != tasks_.rend(); ++task) {
    if (task->scope != TaskScope::kSession || task->dormant) {
      continue;
    }
    const bool missing_publication =
        task->publishes.has_value() && !state.objects().Find(*task->publishes).has_value();
    if (missing_publication || replay_session_tasks.count(task->id.value) != 0) {
      replay_session_tasks.insert(task->id.value);
      for (const TaskId dependency : task->dependencies) {
        replay_session_tasks.insert(dependency.value);
      }
    }
  }

  for (const TaskDescriptor &task : tasks_) {
    if (!task.dormant && task.scope == TaskScope::kSession) {
      session.emplace(task.id.value, state.RequestSessionTask(
                                         task.id, replay_session_tasks.count(task.id.value) != 0));
    }
  }

  const uint64_t readiness_epoch = state.readiness_epoch();
  bool hot_path = false;
  {
    std::lock_guard<std::mutex> lock(state.hot_path_mutex_);
    hot_path = state.hot_path_plan_ == this && state.hot_path_epoch_ == readiness_epoch;
  }
  std::vector<std::shared_ptr<const AllocationHandle>> hot_path_pins;
  if (hot_path) {
    hot_path_pins.reserve(prepared_requirements_.size());
    for (const PreparedKey &key : prepared_requirements_) {
      const std::optional<PreparedObjectView> view = state.objects().Find(key);
      if (!view.has_value()) {
        hot_path = false;
        break;
      }
      hot_path_pins.push_back(view->pin);
    }
    hot_path = hot_path && state.readiness_epoch() == readiness_epoch;
  }
  if (hot_path) {
    PreparedExecutionResult result;
    result.invocation_id = invocation_id;
    result.used_hot_path = true;
    result.diagnostics.reserve(tasks_.size());
    result.session_generations.reserve(session.size());
    const CpuExecutorScope executor_scope(cpu_executor);
    for (const TaskDescriptor &task : tasks_) {
      if (task.dormant) {
        result.diagnostics.push_back({task.id, TaskStatus::kPending, {}, std::nullopt});
        continue;
      }
      if (task.scope == TaskScope::kSession) {
        const auto &request = session.at(task.id.value);
        result.diagnostics.push_back(request.completion.diagnostic());
        result.session_generations.emplace_back(task.id, request.generation);
        continue;
      }
      std::optional<TaskId> failed_dependency;
      for (const TaskId dependency : task.dependencies) {
        const auto found = std::find_if(result.diagnostics.begin(), result.diagnostics.end(),
                                        [dependency](const TaskDiagnostic &diagnostic) {
                                          return diagnostic.task_id == dependency &&
                                                 diagnostic.status != TaskStatus::kSucceeded;
                                        });
        if (found != result.diagnostics.end()) {
          failed_dependency = dependency;
          break;
        }
      }
      if (failed_dependency.has_value()) {
        result.diagnostics.push_back({task.id, TaskStatus::kSuppressed,
                                      "Prepared task dependency did not succeed.",
                                      failed_dependency});
        continue;
      }
      try {
        executor(task, state);
        result.diagnostics.push_back({task.id, TaskStatus::kSucceeded, {}, std::nullopt});
      } catch (...) {
        result.diagnostics.push_back(
            {task.id, TaskStatus::kFailed, "Prepared invocation task failed.", std::nullopt});
      }
    }
    return result;
  }

  std::unordered_map<uint64_t, TaskCompletion> invocation;
  invocation.reserve(tasks_.size() - session.size());
  for (const TaskDescriptor &task : tasks_) {
    if (!task.dormant && task.scope == TaskScope::kInvocation) {
      invocation.emplace(task.id.value, TaskCompletion(task.id));
    }
  }

  struct RunTask {
    const TaskDescriptor *descriptor = nullptr;
    TaskCompletion completion{TaskId{}};
    bool producer = false;
    bool dispatched = false;
    TaskPriority effective_priority = TaskPriority::kBackground;
  };
  std::vector<RunTask> run_tasks;
  run_tasks.reserve(tasks_.size());
  std::unordered_map<uint64_t, size_t> task_indices;
  task_indices.reserve(tasks_.size());
  for (const TaskDescriptor &task : tasks_) {
    if (task.dormant) {
      continue;
    }
    const bool is_session = task.scope == TaskScope::kSession;
    auto request =
        is_session
            ? session.at(task.id.value)
            : PreparedExecutionState::SessionTaskRequest{0, invocation.at(task.id.value), true};
    task_indices.emplace(task.id.value, run_tasks.size());
    run_tasks.push_back(RunTask{&task, request.completion, request.producer, false,
                                is_session ? task.priority : TaskPriority::kCritical});
  }
  for (auto task = run_tasks.rbegin(); task != run_tasks.rend(); ++task) {
    for (const TaskId dependency : task->descriptor->dependencies) {
      RunTask &producer = run_tasks.at(task_indices.at(dependency.value));
      if (PriorityRank(task->effective_priority) > PriorityRank(producer.effective_priority)) {
        producer.effective_priority = task->effective_priority;
      }
    }
  }

  std::mutex progress_mutex;
  std::condition_variable progress;
  size_t active_io = 0;
  size_t progress_epoch = 0;
  size_t peak_in_flight = 0;
  size_t continuation_suspensions = 0;
  auto execute_task = [&](RunTask &task) {
    try {
      TaskDescriptor scheduled = *task.descriptor;
      scheduled.priority = task.effective_priority;
      executor(scheduled, state);
      task.completion.Succeed();
    } catch (...) {
      task.completion.Fail(std::current_exception(), task.descriptor->scope == TaskScope::kSession
                                                         ? "Prepared session task failed."
                                                         : "Prepared invocation task failed.");
    }
    state.scheduler_->Release(*task.descriptor);
  };

  size_t terminal_count = 0;
  while (terminal_count < run_tasks.size()) {
    terminal_count = 0;
    bool critical_pending = false;
    for (const RunTask &task : run_tasks) {
      const TaskStatus status = task.completion.status();
      terminal_count += IsTerminal(status) ? 1 : 0;
      critical_pending |= !IsTerminal(status) && task.effective_priority == TaskPriority::kCritical;
    }
    if (terminal_count == run_tasks.size()) {
      break;
    }

    std::vector<RunTask *> ready;
    bool changed_status = false;
    for (RunTask &task : run_tasks) {
      if (task.dispatched || !task.producer || task.completion.status() != TaskStatus::kPending) {
        continue;
      }
      bool dependencies_ready = true;
      for (const TaskId dependency : task.descriptor->dependencies) {
        const TaskStatus status =
            run_tasks.at(task_indices.at(dependency.value)).completion.status();
        if (IsTerminal(status) && status != TaskStatus::kSucceeded) {
          task.completion.Suppress(dependency, "Prepared task dependency did not succeed.");
          changed_status = true;
          dependencies_ready = false;
          break;
        }
        if (status != TaskStatus::kSucceeded) {
          dependencies_ready = false;
          break;
        }
      }
      if (dependencies_ready) {
        ready.push_back(&task);
      }
    }
    std::stable_sort(ready.begin(), ready.end(), [](const RunTask *left, const RunTask *right) {
      return PriorityRank(left->effective_priority) > PriorityRank(right->effective_priority);
    });

    bool dispatched_any = false;
    std::vector<RunTask *> cpu_tasks;
    for (RunTask *task : ready) {
      TaskDescriptor admitted = *task->descriptor;
      admitted.priority = task->effective_priority;
      size_t total = 0;
      if (!state.scheduler_->TryAcquire(admitted, critical_pending, total)) {
        continue;
      }
      peak_in_flight = std::max(peak_in_flight, total);
      task->dispatched = true;
      task->completion.MarkRunning();
      dispatched_any = true;
      if (task->descriptor->resource == ResourceClass::kIo) {
        {
          std::lock_guard<std::mutex> lock(progress_mutex);
          ++active_io;
        }
        state.scheduler_->io_pool.Enqueue([&, task]() {
          execute_task(*task);
          {
            std::lock_guard<std::mutex> lock(progress_mutex);
            --active_io;
            ++progress_epoch;
            progress.notify_all();
          }
        });
      } else if (task->descriptor->resource == ResourceClass::kCpu && cpu_executor != nullptr) {
        cpu_tasks.push_back(task);
      } else {
        const CpuExecutorScope scope(cpu_executor);
        execute_task(*task);
      }
    }

    if (!cpu_tasks.empty()) {
      struct CpuBatch {
        std::vector<RunTask *> *tasks;
        decltype(execute_task) *execute;
      } batch{&cpu_tasks, &execute_task};
      cpu_executor->ParallelFor(static_cast<int64_t>(cpu_tasks.size()), 1, &batch,
                                [](void *context, int64_t begin, int64_t end) noexcept {
                                  auto &work = *static_cast<CpuBatch *>(context);
                                  for (int64_t i = begin; i < end; ++i) {
                                    (*work.execute)(*work.tasks->at(static_cast<size_t>(i)));
                                  }
                                });
      progress.notify_all();
    }

    if (!dispatched_any) {
      if (changed_status) {
        continue;
      }
      bool waited = false;
      for (RunTask &task : run_tasks) {
        if (!task.producer && !IsTerminal(task.completion.status())) {
          ++continuation_suspensions;
          try {
            task.completion.Wait();
          } catch (...) {
          }
          waited = true;
          break;
        }
      }
      if (!waited) {
        std::unique_lock<std::mutex> lock(progress_mutex);
        if (active_io != 0) {
          ++continuation_suspensions;
          const size_t observed_epoch = progress_epoch;
          progress.wait(lock, [&]() { return progress_epoch != observed_epoch; });
        } else {
          EXT_ENFORCE(false, "Prepared scheduler could not admit a ready task.");
        }
      }
    }
  }
  {
    std::unique_lock<std::mutex> lock(progress_mutex);
    progress.wait(lock, [&]() { return active_io == 0; });
  }

  PreparedExecutionResult result;
  result.invocation_id = invocation_id;
  result.enqueued_tasks = std::count_if(run_tasks.begin(), run_tasks.end(),
                                        [](const RunTask &task) { return task.dispatched; });
  result.continuation_suspensions = continuation_suspensions;
  result.peak_in_flight_bytes = peak_in_flight;
  result.diagnostics.reserve(tasks_.size());
  result.session_generations.reserve(session.size());
  for (const TaskDescriptor &task : tasks_) {
    if (task.dormant) {
      result.diagnostics.push_back({task.id, TaskStatus::kPending, {}, std::nullopt});
      continue;
    }
    const RunTask &run_task = run_tasks.at(task_indices.at(task.id.value));
    result.diagnostics.push_back(run_task.completion.diagnostic());
    if (task.scope == TaskScope::kSession) {
      result.session_generations.emplace_back(task.id, session.at(task.id.value).generation);
    }
  }
  const bool succeeded = std::all_of(run_tasks.begin(), run_tasks.end(), [](const RunTask &task) {
    return task.completion.status() == TaskStatus::kSucceeded;
  });
  if (succeeded && AllPreparedRequirementsResident(tasks_, state.objects())) {
    std::lock_guard<std::mutex> lock(state.hot_path_mutex_);
    state.hot_path_plan_ = this;
    state.hot_path_epoch_ = state.readiness_epoch();
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
                                               size_t prepared_retention_cap,
                                               PreparedSchedulerOptions scheduler_options)
    : preparation_arena_(preparation_slots, preparation_retention_cap),
      prepared_arena_(prepared_slots, prepared_retention_cap),
      scheduler_(std::make_unique<SchedulerState>(std::move(scheduler_options))) {}

PreparedExecutionState::~PreparedExecutionState() = default;

PreparedExecutionState::SessionTaskRequest
PreparedExecutionState::RequestSessionTask(TaskId task_id, bool force_retry) {
  std::lock_guard<std::mutex> lock(session_tasks_mutex_);
  auto [found, inserted] =
      session_tasks_.try_emplace(task_id.value, std::make_unique<SessionTaskEntry>(task_id));
  SessionTaskEntry &entry = *found->second;
  const TaskStatus status = entry.completion->status();
  const bool producer = inserted || (force_retry && status == TaskStatus::kSucceeded) ||
                        status == TaskStatus::kFailed || status == TaskStatus::kCancelled ||
                        status == TaskStatus::kSuppressed;
  if (!inserted && producer) {
    ++entry.generation;
    entry.completion = std::make_shared<TaskCompletion>(task_id);
  }
  return SessionTaskRequest{entry.generation, *entry.completion, producer};
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
