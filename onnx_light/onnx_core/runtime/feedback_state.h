// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/prepared_task.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_proto/onnx_verify.h"
#include <atomic>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

using FeedbackBindings = std::unordered_map<std::string, std::string>;

/**
 * Retains explicitly selected outputs for the next ordinary session invocation.
 *
 * The final model must outlive this state and must not be changed. Each operation
 * checks its serialized snapshot before accessing the session's node pointers.
 * Retained values are isolated from caller-visible outputs. Owned output storage
 * transfers without copying; no execution-arena storage escapes a run.
 * Only tensors, named structures and inline encoded values are supported.
 * The first context's allocators must outlive the state. Later calls must use
 * those same allocators, as required by the retained RuntimeSession kernels.
 */
class ONNX_LIGHT_CORE_API FeedbackState {
public:
  FeedbackState(const ModelProto &model, FeedbackBindings bindings, const RuntimeValueMap &initial,
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
  /** Replaces retained values with explicitly supplied initial contents. */
  void Reset(const RuntimeValueMap &initial);
  /** Releases retained values and kernels and permanently closes this state. */
  void Close();
  /** Returns an independent, owning snapshot of the selected destination paths. */
  RuntimeValueMap Values() const;

private:
  void CheckModel() const;
  RuntimeValueMap ValidateInitial(const RuntimeValueMap &initial) const;

  const ModelProto &model_;
  std::string model_snapshot_;
  StructTypeCatalogue catalogue_;
  FeedbackBindings bindings_;
  std::unique_ptr<RuntimeSession> session_;
  RuntimeValueMap values_;
  bool allocators_captured_ = false;
  RawBufferAllocator *execution_allocator_ = nullptr;
  RawBufferAllocator *io_allocator_ = nullptr;
  mutable std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
