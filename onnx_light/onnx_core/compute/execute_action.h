// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/inplace_reuse_types.h"
#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

/**
 * @file execute_action.h
 * @brief Single step of an :cpp:class:`ExecutionPlan`
 *        (:cpp:class:`ExecuteAction`) describing one memory-management or
 *        node-execution operation.
 */

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Kind of a single :cpp:class:`ExecuteAction`.
 *
 * The values enumerate every operation an :cpp:class:`ExecutionPlan` may
 * schedule while running a graph, function or free-standing node range.
 */
enum class ExecuteActionKind : int32_t {
  /// Locks an initializer so it stays alive while it is still referenced.
  kLockInitializer = 0,
  /// Unlocks an initializer once no remaining node references it.
  kUnlockInitializer = 1,
  /// Locks an input so it stays alive while it is still referenced.
  kLockInput = 2,
  /// Unlocks an input once no remaining node references it.
  kUnlockInput = 3,
  /// Allocates a buffer for a named result.
  kAllocateBuffer = 4,
  /// Deletes a named result (frees its buffer).
  kDeleteBuffer = 5,
  /// Transfers a named result to another named result.
  kTransfer = 6,
  /// Executes a node.
  kExecuteNode = 7,
  /// Creates the shape of a named result.
  kCreateShape = 8,
  /// Deletes the shape of a named result.
  kDeleteShape = 9,
  /// Allocates a temporary buffer required for one or several kernels to
  /// handle a memory peak.
  kAllocateTemporaryBuffer = 10,
  /// Deallocates a temporary buffer once the kernel(s) using it are done.
  kDeleteTemporaryBuffer = 11,
  /// Deletes a named sequence result (frees the sequence it holds).
  kDeleteSequence = 12,
  /// Deletes a named map result (frees the map it holds).
  kDeleteMap = 13,
};

/// Returns a stable, human-readable name for ``kind``.
inline constexpr const char *ExecuteActionKindName(ExecuteActionKind kind) noexcept {
  switch (kind) {
  case ExecuteActionKind::kLockInitializer:
    return "LockInitializer";
  case ExecuteActionKind::kUnlockInitializer:
    return "UnlockInitializer";
  case ExecuteActionKind::kLockInput:
    return "LockInput";
  case ExecuteActionKind::kUnlockInput:
    return "UnlockInput";
  case ExecuteActionKind::kAllocateBuffer:
    return "AllocateBuffer";
  case ExecuteActionKind::kDeleteBuffer:
    return "DeleteBuffer";
  case ExecuteActionKind::kTransfer:
    return "Transfer";
  case ExecuteActionKind::kExecuteNode:
    return "ExecuteNode";
  case ExecuteActionKind::kCreateShape:
    return "CreateShape";
  case ExecuteActionKind::kDeleteShape:
    return "DeleteShape";
  case ExecuteActionKind::kAllocateTemporaryBuffer:
    return "AllocateTemporaryBuffer";
  case ExecuteActionKind::kDeleteTemporaryBuffer:
    return "DeleteTemporaryBuffer";
  case ExecuteActionKind::kDeleteSequence:
    return "DeleteSequence";
  case ExecuteActionKind::kDeleteMap:
    return "DeleteMap";
  }
  return "Unknown";
}

/**
 * Single step of an :cpp:class:`ExecutionPlan`.
 *
 * An action captures exactly one operation the runtime performs while
 * executing a node sequence: locking/unlocking an input or initializer,
 * allocating or deleting a named result (or a temporary buffer), creating
 * or deleting the shape of a named result, transferring a named result to
 * another one, or executing a node.
 */
class ExecuteAction {
public:
  ExecuteAction() = default;

  /**
   * Builds an action.
   *
   * @param kind       Kind of the action.
   * @param name       Primary named result / input / initializer the action
   *                   operates on. Empty for actions that do not target a
   *                   name (e.g. a bare node execution).
   * @param node_index Index of the node for :cpp:enumerator:`kExecuteNode`;
   *                   ``0`` otherwise.
   * @param size       Number of bytes for buffer allocations; ``0`` when
   *                   unknown or not applicable.
   * @param target     Destination named result for
   *                   :cpp:enumerator:`kTransfer`, or the input buffer reused
   *                   by an in-place :cpp:enumerator:`kAllocateBuffer`; empty
   *                   otherwise.
   * @param inplace    In-place reuse decision backing an
   *                   :cpp:enumerator:`kAllocateBuffer` action. The default
   *                   (``output_index < 0``) means the allocation is a fresh
   *                   allocation rather than an in-place reuse.
   */
  ExecuteAction(ExecuteActionKind kind, std::string name, size_t node_index = 0, size_t size = 0,
                std::string target = std::string(),
                compute::InPlaceReuse inplace = compute::InPlaceReuse{})
      : kind_(kind), name_(std::move(name)), target_(std::move(target)), node_index_(node_index),
        size_(size), inplace_(inplace) {}

  /// Returns the kind of the action.
  ExecuteActionKind kind() const noexcept { return kind_; }

  /// Returns the stable, human-readable name of :cpp:func:`kind`.
  const char *kind_name() const noexcept { return ExecuteActionKindName(kind_); }

  /// Returns the primary named result / input / initializer the action
  /// operates on (empty when not applicable).
  const std::string &name() const noexcept { return name_; }

  /// Returns the destination named result of a
  /// :cpp:enumerator:`ExecuteActionKind::kTransfer` action (empty otherwise).
  const std::string &target() const noexcept { return target_; }

  /// Returns the index of the node for
  /// :cpp:enumerator:`ExecuteActionKind::kExecuteNode` (``0`` otherwise).
  size_t node_index() const noexcept { return node_index_; }

  /// Returns the number of bytes for buffer allocations (``0`` when unknown
  /// or not applicable).
  size_t size() const noexcept { return size_; }

  /// Returns ``true`` when this action reuses an input buffer in place rather
  /// than allocating fresh memory (only meaningful for
  /// :cpp:enumerator:`ExecuteActionKind::kAllocateBuffer`).
  bool is_inplace() const noexcept { return inplace_.output_index >= 0; }

  /// Returns the in-place reuse decision backing this action. When
  /// :cpp:func:`is_inplace` is ``false`` the returned value has
  /// ``output_index == -1``.
  const compute::InPlaceReuse &inplace() const noexcept { return inplace_; }

  /**
   * Returns a concise, human-readable one-line summary of the action.
   *
   * The summary starts with the action :cpp:func:`kind_name` and only appends
   * the fields relevant to that kind (node index for node execution, byte size
   * for buffer allocations, transfer destination, in-place reuse). It is meant
   * for logging and debugging an :cpp:class:`ExecutionPlan`.
   *
   * @return A short description such as ``"AllocateBuffer name='y' size=16"``.
   */
  std::string summary() const {
    std::string text = kind_name();
    switch (kind_) {
    case ExecuteActionKind::kExecuteNode:
      text += " node_index=" + std::to_string(node_index_);
      break;
    case ExecuteActionKind::kAllocateBuffer:
    case ExecuteActionKind::kAllocateTemporaryBuffer:
      text += " name='" + name_ + "' size=" + std::to_string(size_);
      if (is_inplace()) {
        text += " inplace(output=" + std::to_string(inplace_.output_index) +
                ", input=" + std::to_string(inplace_.input_index) + ")";
        if (!target_.empty()) {
          text += " reuses='" + target_ + "'";
        }
      }
      break;
    case ExecuteActionKind::kTransfer:
      text += " name='" + name_ + "' -> target='" + target_ + "'";
      break;
    case ExecuteActionKind::kLockInitializer:
    case ExecuteActionKind::kUnlockInitializer:
    case ExecuteActionKind::kLockInput:
    case ExecuteActionKind::kUnlockInput:
    case ExecuteActionKind::kDeleteBuffer:
    case ExecuteActionKind::kDeleteTemporaryBuffer:
    case ExecuteActionKind::kCreateShape:
    case ExecuteActionKind::kDeleteShape:
    case ExecuteActionKind::kDeleteSequence:
    case ExecuteActionKind::kDeleteMap:
      text += " name='" + name_ + "'";
      break;
    }
    return text;
  }

private:
  ExecuteActionKind kind_ = ExecuteActionKind::kExecuteNode;
  std::string name_;
  std::string target_;
  size_t node_index_ = 0;
  size_t size_ = 0;
  compute::InPlaceReuse inplace_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
