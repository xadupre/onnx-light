// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "cuda_gemm_preparation.h"

#include <exception>
#include <stdexcept>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

void AppendIdentityField(std::string &key, const std::string &value) {
  key += std::to_string(value.size());
  key += ':';
  key += value;
  key += ':';
}

const DevicePreparationSubmit &Submitter(CudaGemmPreparationPath path,
                                         const CudaGemmPreparationBackend &backend) {
  return path == CudaGemmPreparationPath::kDevicePack ? backend.device_pack
                                                      : backend.cpu_pack_and_copy;
}

std::string PathName(CudaGemmPreparationPath path) {
  return path == CudaGemmPreparationPath::kDevicePack ? "device-side packing"
                                                      : "CPU packing plus device copy";
}

struct AttemptResult {
  bool succeeded = false;
  uint64_t generation = 0;
  std::string diagnostic;
};

AttemptResult AttemptPreparation(PreparedExecutionState &state,
                                 const PreparedObjectRequirement &requirement,
                                 const DevicePreparationSubmit &submit) {
  PreparedObjectRequest request = state.objects().Request(requirement);
  EXT_ENFORCE(request.producer, "CUDA Gemm preparation unexpectedly joined an in-flight variant.");
  state.objects().MarkPreparing(request);

  DevicePreparationSubmission submission;
  try {
    submission = submit(state);
    EXT_ENFORCE(submission.completion != nullptr,
                "A device preparation submission must return a completion event.");
    submission.completion->Wait();
    EXT_ENFORCE(static_cast<bool>(submission.allocation),
                "A completed device preparation must own an allocation.");
  } catch (const std::exception &error) {
    const std::string diagnostic = error.what();
    state.objects().Fail(request, std::current_exception(), diagnostic);
    return {false, request.generation, diagnostic};
  }

  state.objects().Publish(request, std::move(submission.allocation),
                          std::move(submission.resident_owner));
  return {true, request.generation, {}};
}

} // namespace

PreparedKey CudaGemmVariantIdentity::Key() const {
  EXT_ENFORCE(device_ordinal >= 0, "A CUDA Gemm variant requires a non-negative device ordinal.");
  EXT_ENFORCE(!device_architecture.empty(), "A CUDA Gemm variant requires a device architecture.");
  EXT_ENFORCE(!layout.empty(), "A CUDA Gemm variant requires a layout.");
  EXT_ENFORCE(!kernel_abi.empty(), "A CUDA Gemm variant requires a kernel ABI.");
  EXT_ENFORCE(!source_lineage.empty(), "A CUDA Gemm variant requires source lineage.");

  std::string key = "cuda-gemm:";
  AppendIdentityField(key, std::to_string(device_ordinal));
  AppendIdentityField(key, device_architecture);
  AppendIdentityField(key, layout);
  AppendIdentityField(key, kernel_abi);
  for (const std::string &source : source_lineage) {
    EXT_ENFORCE(!source.empty(), "CUDA Gemm source lineage entries must not be empty.");
    AppendIdentityField(key, source);
  }
  return {std::move(key)};
}

CudaGemmPreparationTasks ExpandCudaGemmPreparation(const CudaGemmVariantIdentity &identity,
                                                   CudaGemmPreparationPath path,
                                                   TaskId first_task_id) {
  EXT_ENFORCE(first_task_id.value != 0, "CUDA Gemm preparation task IDs must not be zero.");
  const PreparedKey key = identity.Key();
  const TaskId load_id{first_task_id.value};
  const TaskId transform_id{first_task_id.value + 1};
  const TaskId device_id{first_task_id.value + 2};
  const TaskId publish_id{first_task_id.value + 3};

  CudaGemmPreparationTasks result;
  result.path = path;
  result.tasks.push_back(
      {load_id, TaskScope::kSession, TaskKind::kReadPayload, ResourceClass::kIo});
  result.tasks.back().payload_id = identity.source_lineage.front();
  if (path == CudaGemmPreparationPath::kCpuPackAndCopy) {
    result.tasks.push_back(
        {transform_id, TaskScope::kSession, TaskKind::kPrepare, ResourceClass::kCpu, {load_id}});
    result.tasks.push_back(
        {device_id, TaskScope::kSession, TaskKind::kCopy, ResourceClass::kDevice, {transform_id}});
  } else {
    result.tasks.push_back(
        {transform_id, TaskScope::kSession, TaskKind::kCopy, ResourceClass::kDevice, {load_id}});
    result.tasks.push_back({device_id,
                            TaskScope::kSession,
                            TaskKind::kPrepare,
                            ResourceClass::kDevice,
                            {transform_id}});
  }
  result.tasks.push_back(
      {publish_id, TaskScope::kSession, TaskKind::kPublish, ResourceClass::kInline, {device_id}});
  result.tasks.back().publishes = key;
  result.tasks.back().payload_id = identity.source_lineage.front();
  return result;
}

CudaGemmPreparationResult PrepareCudaGemmVariant(PreparedExecutionState &state,
                                                 const CudaGemmVariantIdentity &identity,
                                                 const CudaGemmPreparationPolicy &policy,
                                                 const CudaGemmPreparationBackend &backend) {
  const PreparedKey key = identity.Key();
  const PreparedObjectRequirement requirement{key, identity.source_lineage.front()};
  const CudaGemmPreparationPath preferred = policy.prefer_device_pack
                                                ? CudaGemmPreparationPath::kDevicePack
                                                : CudaGemmPreparationPath::kCpuPackAndCopy;
  const CudaGemmPreparationPath alternative = preferred == CudaGemmPreparationPath::kDevicePack
                                                  ? CudaGemmPreparationPath::kCpuPackAndCopy
                                                  : CudaGemmPreparationPath::kDevicePack;

  const DevicePreparationSubmit &preferred_submit = Submitter(preferred, backend);
  const DevicePreparationSubmit &alternative_submit = Submitter(alternative, backend);
  EXT_ENFORCE(static_cast<bool>(preferred_submit) || static_cast<bool>(alternative_submit),
              "No CUDA Gemm preparation path is supported.");

  CudaGemmPreparationResult result;
  result.identity = identity;
  result.selected_path = preferred;
  if (preferred_submit) {
    const AttemptResult attempt = AttemptPreparation(state, requirement, preferred_submit);
    if (attempt.succeeded) {
      result.generation = attempt.generation;
      return result;
    }
    result.used_fallback = true;
    result.fallback_diagnostic = PathName(preferred) + " failed: " + attempt.diagnostic;
  } else {
    result.used_fallback = true;
    result.fallback_diagnostic = PathName(preferred) + " is unavailable";
  }

  EXT_ENFORCE(static_cast<bool>(alternative_submit), result.fallback_diagnostic,
              "; no alternative CUDA Gemm preparation path is supported.");
  const AttemptResult fallback = AttemptPreparation(state, requirement, alternative_submit);
  EXT_ENFORCE(fallback.succeeded, result.fallback_diagnostic, "; ", PathName(alternative),
              " failed: ", fallback.diagnostic);
  result.selected_path = alternative;
  result.generation = fallback.generation;
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
