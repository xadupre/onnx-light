// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/prepared_execution.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

enum class CudaGemmPreparationPath {
  kCpuPackAndCopy,
  kDevicePack,
};

struct CudaGemmVariantIdentity {
  int32_t device_ordinal = 0;
  std::string device_architecture;
  std::string layout;
  std::string kernel_abi;
  std::vector<std::string> source_lineage;

  PreparedKey Key() const;
};

struct CudaGemmPreparationPolicy {
  bool prefer_device_pack = false;
};

struct CudaGemmPreparationTasks {
  CudaGemmPreparationPath path = CudaGemmPreparationPath::kCpuPackAndCopy;
  std::vector<TaskDescriptor> tasks;
};

ONNX_LIGHT_CORE_API CudaGemmPreparationTasks ExpandCudaGemmPreparation(
    const CudaGemmVariantIdentity &identity, CudaGemmPreparationPath path, TaskId first_task_id);

class DevicePreparationEvent {
public:
  virtual ~DevicePreparationEvent() = default;
  virtual void Wait() = 0;
};

struct DevicePreparationSubmission {
  AllocationHandle allocation;
  std::shared_ptr<DevicePreparationEvent> completion;
  std::vector<std::shared_ptr<void>> submission_owners;
  std::shared_ptr<void> resident_owner;
};

using DevicePreparationSubmit =
    std::function<DevicePreparationSubmission(PreparedExecutionState &)>;

struct CudaGemmPreparationBackend {
  DevicePreparationSubmit cpu_pack_and_copy;
  DevicePreparationSubmit device_pack;
};

struct CudaGemmPreparationResult {
  CudaGemmVariantIdentity identity;
  CudaGemmPreparationPath selected_path = CudaGemmPreparationPath::kCpuPackAndCopy;
  bool used_fallback = false;
  std::string fallback_diagnostic;
  uint64_t generation = 0;
};

/**
 * Prepares and atomically publishes one fixed-placement CUDA Gemm variant.
 *
 * The selected backend submission must return an explicit completion event.
 * Submission owners remain alive through that event, and the resident owner
 * remains attached to the published object. A failed preferred path is marked
 * failed before a new generation is produced by the alternative path.
 */
ONNX_LIGHT_CORE_API CudaGemmPreparationResult PrepareCudaGemmVariant(
    PreparedExecutionState &state, const CudaGemmVariantIdentity &identity,
    const CudaGemmPreparationPolicy &policy, const CudaGemmPreparationBackend &backend);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
