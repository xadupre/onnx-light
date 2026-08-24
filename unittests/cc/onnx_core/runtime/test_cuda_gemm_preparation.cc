// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/cuda_gemm_preparation.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

CudaGemmVariantIdentity TestIdentity() {
  return {0,
          "sm_80",
          "gemm-transB=1-tile=128x64",
          "onnx-light.cuda.gemm.v1",
          {"model-digest", "B-digest"}};
}

class TestDeviceEvent final : public DevicePreparationEvent {
public:
  TestDeviceEvent(std::weak_ptr<void> submission_owner, bool fail, bool *waited)
      : submission_owner_(std::move(submission_owner)), fail_(fail), waited_(waited) {}

  void Wait() override {
    *waited_ = true;
    EXPECT_FALSE(submission_owner_.expired());
    if (fail_) {
      throw std::runtime_error("device event failed");
    }
  }

private:
  std::weak_ptr<void> submission_owner_;
  bool fail_;
  bool *waited_;
};

DevicePreparationSubmission TestSubmission(PreparedExecutionState &state, bool fail, bool *waited,
                                           const std::shared_ptr<void> &submission_owner,
                                           std::shared_ptr<void> resident_owner = {}) {
  DevicePreparationSubmission submission;
  submission.allocation =
      AllocationHandle(&state.prepared_arena(), state.prepared_arena().Allocate(4));
  submission.completion = std::make_shared<TestDeviceEvent>(submission_owner, fail, waited);
  submission.submission_owners.push_back(submission_owner);
  submission.resident_owner = std::move(resident_owner);
  return submission;
}

} // namespace

TEST(CudaGemmPreparation, VariantIdentityIncludesDeviceLayoutAbiAndLineage) {
  const CudaGemmVariantIdentity identity = TestIdentity();
  const std::string key = identity.Key().value;

  CudaGemmVariantIdentity changed = identity;
  changed.device_ordinal = 1;
  EXPECT_NE(changed.Key(), identity.Key());
  changed = identity;
  changed.layout += "-other";
  EXPECT_NE(changed.Key(), identity.Key());
  changed = identity;
  changed.kernel_abi += "-other";
  EXPECT_NE(changed.Key(), identity.Key());
  changed = identity;
  changed.source_lineage[1] += "-other";
  EXPECT_NE(changed.Key(), identity.Key());
  EXPECT_NE(key.find("sm_80"), std::string::npos);
}

TEST(CudaGemmPreparation, ExpandsExplicitCpuCopyAndDevicePackResources) {
  const CudaGemmVariantIdentity identity = TestIdentity();
  const CudaGemmPreparationTasks cpu =
      ExpandCudaGemmPreparation(identity, CudaGemmPreparationPath::kCpuPackAndCopy, TaskId{10});
  const CudaGemmPreparationTasks device =
      ExpandCudaGemmPreparation(identity, CudaGemmPreparationPath::kDevicePack, TaskId{20});

  ASSERT_EQ(cpu.tasks.size(), 4u);
  EXPECT_EQ(cpu.tasks[0].resource, ResourceClass::kIo);
  EXPECT_EQ(cpu.tasks[1].resource, ResourceClass::kCpu);
  EXPECT_EQ(cpu.tasks[1].kind, TaskKind::kPrepare);
  EXPECT_EQ(cpu.tasks[2].resource, ResourceClass::kDevice);
  EXPECT_EQ(cpu.tasks[2].kind, TaskKind::kCopy);
  EXPECT_EQ(cpu.tasks[3].dependencies, (std::vector<TaskId>{TaskId{12}}));
  EXPECT_EQ(cpu.tasks[3].publishes, identity.Key());

  ASSERT_EQ(device.tasks.size(), 4u);
  EXPECT_EQ(device.tasks[1].resource, ResourceClass::kDevice);
  EXPECT_EQ(device.tasks[1].kind, TaskKind::kCopy);
  EXPECT_EQ(device.tasks[2].resource, ResourceClass::kDevice);
  EXPECT_EQ(device.tasks[2].kind, TaskKind::kPrepare);
}

TEST(CudaGemmPreparation, UnsupportedPreferredPathFallsBackExplicitly) {
  PreparedExecutionState state;
  const CudaGemmVariantIdentity identity = TestIdentity();
  bool waited = false;
  CudaGemmPreparationBackend backend;
  backend.cpu_pack_and_copy = [&](PreparedExecutionState &run_state) {
    const auto owner = std::make_shared<int>(1);
    return TestSubmission(run_state, false, &waited, owner);
  };

  const CudaGemmPreparationResult result =
      PrepareCudaGemmVariant(state, identity, {.prefer_device_pack = true}, backend);

  EXPECT_TRUE(result.used_fallback);
  EXPECT_EQ(result.selected_path, CudaGemmPreparationPath::kCpuPackAndCopy);
  EXPECT_NE(result.fallback_diagnostic.find("unavailable"), std::string::npos);
  EXPECT_TRUE(waited);
  EXPECT_EQ(state.objects().State(identity.Key()), PreparedResidencyState::kResident);
}

TEST(CudaGemmPreparation, FailedVariantPublishesOnlySuccessfulFallbackGeneration) {
  PreparedExecutionState state;
  const CudaGemmVariantIdentity identity = TestIdentity();
  bool device_waited = false;
  bool fallback_waited = false;
  CudaGemmPreparationBackend backend;
  backend.device_pack = [&](PreparedExecutionState &run_state) {
    const auto owner = std::make_shared<int>(1);
    return TestSubmission(run_state, true, &device_waited, owner);
  };
  backend.cpu_pack_and_copy = [&](PreparedExecutionState &run_state) {
    const auto owner = std::make_shared<int>(2);
    return TestSubmission(run_state, false, &fallback_waited, owner);
  };

  const CudaGemmPreparationResult result =
      PrepareCudaGemmVariant(state, identity, {.prefer_device_pack = true}, backend);
  const std::optional<PreparedObjectView> view = state.objects().Find(identity.Key());

  EXPECT_TRUE(result.used_fallback);
  EXPECT_EQ(result.generation, 2u);
  EXPECT_TRUE(device_waited);
  EXPECT_TRUE(fallback_waited);
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->generation, 2u);
  EXPECT_EQ(state.prepared_arena().allocated_count(), 1u);
}

TEST(CudaGemmPreparation, SubmissionAndResidentOwnersSurviveRequiredLifetimes) {
  PreparedExecutionState state;
  const CudaGemmVariantIdentity identity = TestIdentity();
  bool waited = false;
  std::weak_ptr<void> submission_lifetime;
  std::weak_ptr<void> resident_lifetime;
  CudaGemmPreparationBackend backend;
  backend.device_pack = [&](PreparedExecutionState &run_state) {
    const auto submission_owner = std::make_shared<int>(1);
    const auto resident_owner = std::make_shared<int>(2);
    submission_lifetime = submission_owner;
    resident_lifetime = resident_owner;
    return TestSubmission(run_state, false, &waited, submission_owner, resident_owner);
  };

  PrepareCudaGemmVariant(state, identity, {.prefer_device_pack = true}, backend);
  EXPECT_TRUE(waited);
  EXPECT_TRUE(submission_lifetime.expired());
  EXPECT_FALSE(resident_lifetime.expired());

  std::optional<PreparedObjectView> pin = state.objects().Find(identity.Key());
  ASSERT_TRUE(pin.has_value());
  EXPECT_TRUE(state.objects().Evict(identity.Key()));
  EXPECT_FALSE(resident_lifetime.expired());
  pin.reset();
  EXPECT_TRUE(resident_lifetime.expired());
}
