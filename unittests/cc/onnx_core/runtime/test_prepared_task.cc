// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/prepared_execution.h"
#include "onnx_core/compute/prepared_task.h"
#include "onnx_core/compute/resolved_model_fixture.h"
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>

using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

TEST(PreparedTask, DescriptorCarriesSchedulingMetadata) {
  TaskDescriptor descriptor{
      TaskId{3},
      TaskScope::kSession,
      TaskKind::kReadPayload,
      ResourceClass::kIo,
      {TaskId{1}, TaskId{2}},
      16,
      32,
      8,
      ActionRange{4, 7},
  };

  EXPECT_EQ(descriptor.id.value, 3);
  EXPECT_EQ(descriptor.dependencies.size(), 2);
  ASSERT_TRUE(descriptor.actions.has_value());
  EXPECT_EQ(descriptor.actions->begin, 4);
  EXPECT_EQ(descriptor.actions->end, 7);
}

TEST(PreparedTask, CompletionWaitsAndSharesSuccess) {
  TaskCompletion completion(TaskId{4});
  TaskCompletion observer = completion;
  std::thread producer([completion]() {
    completion.MarkRunning();
    completion.Succeed();
  });

  observer.Wait();
  producer.join();
  EXPECT_TRUE(observer.IsReady());
  EXPECT_EQ(observer.status(), TaskStatus::kSucceeded);
}

TEST(PreparedTask, CompletionPreservesFailureDiagnosticAndException) {
  TaskCompletion completion(TaskId{5});
  completion.Fail(std::make_exception_ptr(std::runtime_error("prepare failed")),
                  "failed to prepare W");

  EXPECT_THROW(completion.Wait(), std::runtime_error);
  TaskDiagnostic diagnostic = completion.diagnostic();
  EXPECT_EQ(diagnostic.task_id.value, 5);
  EXPECT_EQ(diagnostic.status, TaskStatus::kFailed);
  EXPECT_EQ(diagnostic.message, "failed to prepare W");
}

TEST(PreparedTask, CompletionRecordsSuppressedDependency) {
  TaskCompletion completion(TaskId{6});
  completion.Suppress(TaskId{5}, "dependency failed");

  completion.Wait();
  TaskDiagnostic diagnostic = completion.diagnostic();
  EXPECT_EQ(diagnostic.status, TaskStatus::kSuppressed);
  ASSERT_TRUE(diagnostic.caused_by.has_value());
  EXPECT_EQ(diagnostic.caused_by->value, 5);
}

TEST(ResolvedModelFixture, ReadsOnlyActivePayloadManifestEntries) {
  const std::filesystem::path path = "resolved_model_fixture.data";
  {
    std::ofstream stream(path, std::ios::binary);
    stream.write("01234567", 8);
  }

  ResolvedModelFixture resolved("resolved_model_fixture.onnx",
                                {
                                    PayloadManifestEntry{"active", path, 2, 4, true},
                                    PayloadManifestEntry{"inactive", path, 0, 2, false},
                                });

  EXPECT_EQ(resolved.ReadPayload("active"), (std::vector<uint8_t>{'2', '3', '4', '5'}));
  EXPECT_THROW(resolved.ReadPayload("inactive"), std::runtime_error);
  EXPECT_THROW(resolved.ReadPayload("missing"), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(PreparedObjectStore, SharesOneInFlightGenerationAndPublishesAtomically) {
  PreparedExecutionState state(1, 1);
  const PreparedObjectRequirement requirement{PreparedKey{"gemm:B"}, "B"};
  PreparedObjectRequest producer = state.objects().Request(requirement);
  PreparedObjectRequest observer = state.objects().Request(requirement);

  EXPECT_TRUE(producer.producer);
  EXPECT_FALSE(observer.producer);
  EXPECT_EQ(observer.generation, producer.generation);
  EXPECT_FALSE(state.objects().Find(requirement.key).has_value());

  state.objects().MarkPreparing(producer);
  AllocationHandle allocation(&state.prepared_arena(), state.prepared_arena().Allocate(4));
  std::memcpy(allocation.buffer()->data(), "done", 4);
  std::atomic<bool> observed{false};
  std::thread waiter([&]() {
    observer.completion.Wait();
    const std::optional<PreparedObjectView> view = state.objects().Find(requirement.key);
    observed = view.has_value() && std::memcmp(view->buffer->data(), "done", 4) == 0;
  });
  state.objects().Publish(producer, std::move(allocation));
  waiter.join();

  EXPECT_TRUE(observed);
  EXPECT_EQ(state.prepared_arena().allocated_count(), 1u);
}

TEST(PreparedObjectStore, EvictionReturnsAllocationToItsOriginalArena) {
  PreparedExecutionState state(1, 1);
  const PreparedObjectRequirement requirement{PreparedKey{"resident"}, "source"};
  PreparedObjectRequest request = state.objects().Request(requirement);
  AllocationHandle allocation(&state.prepared_arena(), state.prepared_arena().Allocate(8));
  state.objects().Publish(request, std::move(allocation));

  std::optional<PreparedObjectView> view = state.objects().Find(requirement.key);
  ASSERT_TRUE(view.has_value());
  EXPECT_EQ(view->owner, &state.prepared_arena());
  EXPECT_TRUE(state.objects().Evict(requirement.key));
  EXPECT_EQ(state.prepared_arena().allocated_count(), 1u);
  view.reset();
  EXPECT_EQ(state.prepared_arena().allocated_count(), 0u);
  EXPECT_EQ(state.objects().State(requirement.key), PreparedResidencyState::kAbsent);
}

TEST(PreparedExecutionState, PreparationScratchReturnsToPreparationArena) {
  PreparedExecutionState state(1, 1);
  {
    AllocationHandle scratch(&state.preparation_arena(), state.preparation_arena().Allocate(16));
    EXPECT_EQ(scratch.owner(), &state.preparation_arena());
    EXPECT_EQ(state.preparation_arena().allocated_count(), 1u);
  }
  EXPECT_EQ(state.preparation_arena().allocated_count(), 0u);
  EXPECT_EQ(state.prepared_arena().allocated_count(), 0u);
}

TEST(MaterializationRecipe, ExpandsSessionLoadPrepackPublishAndDormantFallback) {
  PreparedRequirementDescriptor requirement{
      PreparedObjectRequirement{PreparedKey{"gemm:B:packed"}, "B"},
      {MaterializationRecipe{MaterializationRecipeKind::kReadSourceAndPrepack, "B",
                             "gemm-transB=0"}}};
  const MaterializationTaskDescriptors tasks =
      ExpandMaterializationRecipe(requirement, requirement.recipes.front(), TaskId{10});

  EXPECT_EQ(tasks.load.scope, TaskScope::kSession);
  EXPECT_EQ(tasks.load.kind, TaskKind::kReadPayload);
  EXPECT_EQ(tasks.prepack.dependencies, (std::vector<TaskId>{TaskId{10}}));
  EXPECT_EQ(tasks.publish.dependencies, (std::vector<TaskId>{TaskId{11}}));
  ASSERT_TRUE(tasks.publish.publishes.has_value());
  EXPECT_EQ(tasks.publish.publishes->value, "gemm:B:packed");
  EXPECT_EQ(tasks.dormant_fallback.kind, TaskKind::kFallback);
  EXPECT_TRUE(tasks.dormant_fallback.dormant);
}
