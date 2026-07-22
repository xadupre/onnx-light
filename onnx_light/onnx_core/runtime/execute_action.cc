// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/execute_action.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

const char *ExecuteActionKindName(ExecuteActionKind kind) noexcept {
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
  }
  return "Unknown";
}

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
