// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/prepared_task.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_proto/onnx_verify.h"
#include <atomic>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Carries graph-declared persistent values between model calls.
 *
 * Each Run:
 * 1. Binds retained values to persistent inputs and adds the current feeds.
 * 2. Executes the session, reusing eligible append buffers or allocating new ones.
 * 3. Validates the outputs and checks cancellation.
 * 4. Publishes selected outputs as the next state only on success.
 *
 * Storage:
 * - Views share tensor data but have separate names and shapes.
 * - In-place append writes only the new region; growth may copy the old prefix.
 * - Failure leaves the previous logical state unchanged.
 *
 * Requirements:
 * - Each binding names a whole input/output pair; the input has exactly one value-use.
 * - Current feeds cannot replace persistent inputs.
 * - Supports tensors, named structures and inline encoded values with retained storage owners.
 * - Shared payloads are read-only to callers; ownerless borrows are rejected.
 * - The model stays immutable. All calls use the same allocators, kept alive by the caller.
 */
class ONNX_LIGHT_CORE_API PersistentValueState {
public:
  /**
   * Initializes retained values from the graph's persistent bindings.
   *
   * Accepts std::move(initial) to avoid copying owned payloads.
   * Without model_owner, the caller keeps the model alive through the state and exported views.
   */
  PersistentValueState(const ModelProto &model, RuntimeValueMap initial,
                       RuntimeSessionOptions options = {}, std::shared_ptr<void> model_owner = {});
  /** Initializes retained values and keeps the shared model alive. */
  PersistentValueState(std::shared_ptr<const ModelProto> model, RuntimeValueMap initial,
                       RuntimeSessionOptions options = {});

  /**
   * Runs with current feeds and atomically replaces the selected retained values.
   *
   * An optional pending completion acts as a cancellation/commit gate. Cancelling
   * it before publication aborts feedback; success makes publication irrevocable.
   * The token is single-use and must not be marked running by the caller.
   * When context.events_enabled() is true, invocation events are appended to
   * context.events(), including work preceding a failure or cancellation.
   */
  RuntimeValueMap Run(RuntimeContext &context, const RuntimeValueMap &feeds,
                      const TaskCompletion *completion = nullptr);
  /**
   * Retains an external model/context owner until kernels and state are released.
   *
   * The caller registers owners before Run, including attempts that may fail during initialization.
   * Repeated calls with the same shared owner are deduplicated. The token must
   * not own this PersistentValueState, which would create a lifetime cycle.
   */
  void RetainOwner(std::shared_ptr<void> owner);
  /** Replaces retained values with explicitly transferred initial contents. */
  void Reset(RuntimeValueMap initial);
  /** Releases retained values and kernels and permanently closes this state. */
  void Close();
  /** Returns read-only aliases keyed by exact retained graph input names. */
  RuntimeValueMap Values() const;

private:
  struct Binding {
    std::string input;
    std::string output;
    const TypeProto *input_type;
  };
  std::vector<PersistentValue> ValidateInitial(RuntimeValueMap initial) const;

  const ModelProto &model_;
  std::shared_ptr<void> model_owner_;
  std::vector<std::shared_ptr<void>> retained_owners_;
  StructTypeCatalogue catalogue_;
  std::vector<Binding> bindings_;
  std::unique_ptr<RuntimeSession> session_;
  std::vector<PersistentValue> values_;
  size_t persistent_tensor_initial_capacity_ = 0;
  bool allocators_captured_ = false;
  RawBufferAllocator *execution_allocator_ = nullptr;
  RawBufferAllocator *io_allocator_ = nullptr;
  mutable std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
