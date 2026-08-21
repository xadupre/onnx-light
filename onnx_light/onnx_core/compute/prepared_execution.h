// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/prepared_task.h"
#include "onnx_core/compute/raw_buffer_allocator.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
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
      size_t prepared_retention_cap = std::numeric_limits<size_t>::max());

  PreparationArena &preparation_arena() noexcept { return preparation_arena_; }
  PreparedArena &prepared_arena() noexcept { return prepared_arena_; }
  PreparedObjectStore &objects() noexcept { return objects_; }
  const PreparedObjectStore &objects() const noexcept { return objects_; }
  uint64_t readiness_epoch() const noexcept { return objects_.readiness_epoch(); }

private:
  PreparationArena preparation_arena_;
  PreparedArena prepared_arena_;
  PreparedObjectStore objects_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
