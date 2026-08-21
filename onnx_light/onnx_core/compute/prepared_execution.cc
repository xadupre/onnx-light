// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "prepared_execution.h"

#include <functional>
#include <stdexcept>
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

PreparedExecutionState::PreparedExecutionState(size_t preparation_slots, size_t prepared_slots,
                                               size_t preparation_retention_cap,
                                               size_t prepared_retention_cap)
    : preparation_arena_(preparation_slots, preparation_retention_cap),
      prepared_arena_(prepared_slots, prepared_retention_cap) {}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
