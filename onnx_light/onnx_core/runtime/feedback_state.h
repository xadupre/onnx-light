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
 * Retains graph-declared outputs for the next ordinary session invocation.
 *
 * Each binding selects one whole input and one whole output by exact graph name.
 * Structured values retain all fields; current feeds cannot override retained inputs.
 * The final model must not be changed. A reference-only caller must keep it alive
 * until this state and all model-backed output views are released. The shared-model
 * overload or an explicit model_owner token retains that lifetime automatically.
 * Each operation
 * uses cached declaration pointers without cloning or checking the model.
 * Initial values, feeds, returned outputs and Values() share read-only payloads.
 * Callers and kernels must not mutate shared storage. Ownerless borrows and
 * execution-arena allocations lacking self-owning leases are rejected.
 * Only tensors, named structures and inline encoded values are supported.
 * The first context's allocators must outlive the state. Later calls must use
 * those same allocators, as required by the retained RuntimeSession kernels.
 */
class ONNX_LIGHT_CORE_API FeedbackState {
public:
  FeedbackState(const ModelProto &model, const RuntimeValueMap &initial,
                RuntimeSessionOptions options = {}, std::shared_ptr<void> model_owner = {});
  FeedbackState(std::shared_ptr<const ModelProto> model, const RuntimeValueMap &initial,
                RuntimeSessionOptions options = {});

  /**
   * Runs with current feeds and atomically replaces the selected retained values.
   *
   * An optional pending completion acts as a cancellation/commit gate. Cancelling
   * it before publication aborts feedback; success makes publication irrevocable.
   * The token is single-use and must not be marked running by the caller.
   */
  RuntimeValueMap Run(RuntimeContext &context, const RuntimeValueMap &feeds,
                      const TaskCompletion *completion = nullptr);
  /**
   * Retains an external model/context owner until kernels and state are released.
   *
   * The caller registers owners before Run, including attempts that may fail during initialization.
   * Repeated calls with the same shared owner are deduplicated. The token must
   * not own this FeedbackState, which would create a lifetime cycle.
   */
  void RetainOwner(std::shared_ptr<void> owner);
  /** Replaces retained values with explicitly supplied initial contents. */
  void Reset(const RuntimeValueMap &initial);
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
  std::vector<RuntimeValue> ValidateInitial(const RuntimeValueMap &initial) const;

  const ModelProto &model_;
  std::shared_ptr<void> model_owner_;
  std::vector<std::shared_ptr<void>> retained_owners_;
  StructTypeCatalogue catalogue_;
  std::vector<Binding> bindings_;
  std::unique_ptr<RuntimeSession> session_;
  std::vector<RuntimeValue> values_;
  bool allocators_captured_ = false;
  RawBufferAllocator *execution_allocator_ = nullptr;
  RawBufferAllocator *io_allocator_ = nullptr;
  mutable std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
