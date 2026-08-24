// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/prepared_execution.h"
#include "onnx_core/compute/prepared_task.h"
#include "onnx_core/compute/prepared_tensor_cache.h"
#include "onnx_core/compute/resolved_model_fixture.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

PreparedTensorMetadata TestPreparedMetadata() {
  PreparedTensorMetadata metadata;
  metadata.source_digest = "source-digest";
  metadata.architecture = "x86_64";
  metadata.runtime = "onnx-light";
  metadata.runtime_version = "1";
  metadata.kernel_layout = "gemm-transB=0";
  metadata.format_version = "1";
  return metadata;
}

void PopulatePreparedCache(const std::filesystem::path &path,
                           const PreparedTensorMetadata &metadata) {
  PreparedTensorCache cache;
  cache.LoadOrPrepare(
      path, metadata, []() { return std::vector<uint8_t>{1, 2, 3}; },
      [](const std::vector<uint8_t> &source) {
        return std::vector<uint8_t>{source[2], source[1], source[0]};
      },
      [](const std::vector<uint8_t> &) {});
  cache.WaitForBackgroundWrites();
}

} // namespace

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

  ResolvedModelFixture resolved(
      "resolved_model_fixture.onnx",
      RequiredPayloadManifest::Freeze({
          PayloadManifestEntry{"active", path, 2, 4},
          PayloadManifestEntry{"inactive", path, 0, 2, PayloadResolution::kDead},
      }));

  EXPECT_EQ(resolved.ReadPayload("active"), (std::vector<uint8_t>{'2', '3', '4', '5'}));
  EXPECT_THROW(resolved.ReadPayload("inactive"), std::runtime_error);
  EXPECT_THROW(resolved.ReadPayload("missing"), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(RequiredPayloadManifest, FreezesLivePayloadsAndReportsAvoidedBytes) {
  RequiredPayloadManifest manifest = RequiredPayloadManifest::Freeze({
      PayloadManifestEntry{"if_else", "if.bin", 0, 11, PayloadResolution::kDead},
      PayloadManifestEntry{"debug_head", "debug.bin", 0, 13, PayloadResolution::kDead},
      PayloadManifestEntry{"rewritten", "rewrite.bin", 0, 17, PayloadResolution::kSuperseded},
      PayloadManifestEntry{"portable", "weights.bin", 0, 19,
                           PayloadResolution::kPreparedCacheReplaced},
      PayloadManifestEntry{"packed", "cache.bin", 0, 7},
      PayloadManifestEntry{"portable_fallback", "weights.bin", 0, 19,
                           PayloadResolution::kDormantFallback},
  });

  EXPECT_TRUE(manifest.ContainsActive("packed"));
  EXPECT_FALSE(manifest.ContainsActive("if_else"));
  EXPECT_FALSE(manifest.ContainsActive("debug_head"));
  EXPECT_FALSE(manifest.ContainsActive("rewritten"));
  EXPECT_FALSE(manifest.ContainsActive("portable"));
  EXPECT_FALSE(manifest.ContainsActive("portable_fallback"));
  EXPECT_EQ(manifest.avoided().dead_bytes, 24u);
  EXPECT_EQ(manifest.avoided().superseded_bytes, 17u);
  EXPECT_EQ(manifest.avoided().cache_replaced_bytes, 19u);
  EXPECT_EQ(manifest.avoided().total_bytes(), 60u);
}

TEST(RequiredPayloadManifest, DiagnosesEagerFallbackForUnsupportedTransformation) {
  RequiredPayloadManifest manifest = RequiredPayloadManifest::Freeze(
      {
          PayloadManifestEntry{"live", "weights.bin", 0, 7},
          PayloadManifestEntry{"otherwise_dead", "weights.bin", 7, 11, PayloadResolution::kDead},
      },
      "payload-dependent graph rewrite");

  EXPECT_TRUE(manifest.uses_eager_fallback());
  EXPECT_EQ(*manifest.eager_fallback_diagnostic(), "payload-dependent graph rewrite");
  EXPECT_TRUE(manifest.ContainsActive("live"));
  EXPECT_TRUE(manifest.ContainsActive("otherwise_dead"));
  EXPECT_EQ(manifest.avoided().total_bytes(), 0u);
}

TEST(ResolvedModelFixture, ReadTasksMustNameAnActiveFrozenManifestEntry) {
  const std::filesystem::path path = "resolved_model_task_fixture.data";
  {
    std::ofstream stream(path, std::ios::binary);
    stream.write("0123", 4);
  }
  ResolvedModelFixture resolved(
      "resolved_model_fixture.onnx",
      RequiredPayloadManifest::Freeze({PayloadManifestEntry{"live", path, 1, 2}}));
  TaskDescriptor active{TaskId{1}, TaskScope::kSession, TaskKind::kReadPayload, ResourceClass::kIo};
  active.payload_id = "live";
  TaskDescriptor absent{TaskId{2}, TaskScope::kSession, TaskKind::kReadPayload, ResourceClass::kIo};
  absent.payload_id = "absent";
  TaskDescriptor unnamed{TaskId{3}, TaskScope::kSession, TaskKind::kReadPayload,
                         ResourceClass::kIo};

  EXPECT_EQ(resolved.ReadPayload(active), (std::vector<uint8_t>{'1', '2'}));
  EXPECT_THROW(resolved.ReadPayload(absent), std::runtime_error);
  EXPECT_THROW(resolved.ReadPayload(unnamed), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(PreparedExecutionPlan, RejectsReadsOutsideFrozenManifest) {
  const RequiredPayloadManifest manifest =
      RequiredPayloadManifest::Freeze({PayloadManifestEntry{"live", "weights.bin", 0, 4}});
  TaskDescriptor live{TaskId{1}, TaskScope::kSession, TaskKind::kReadPayload, ResourceClass::kIo};
  live.payload_id = "live";
  TaskDescriptor absent{TaskId{2}, TaskScope::kSession, TaskKind::kReadPayload, ResourceClass::kIo};
  absent.payload_id = "absent";
  TaskDescriptor unnamed{TaskId{3}, TaskScope::kSession, TaskKind::kReadPayload,
                         ResourceClass::kIo};

  EXPECT_NO_THROW(PreparedExecutionPlan({live}, manifest));
  EXPECT_THROW(PreparedExecutionPlan({absent}, manifest), std::runtime_error);
  EXPECT_THROW(PreparedExecutionPlan({unnamed}, manifest), std::runtime_error);
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
  EXPECT_EQ(tasks.load.payload_id, "B");
  EXPECT_EQ(tasks.prepack.dependencies, (std::vector<TaskId>{TaskId{10}}));
  EXPECT_EQ(tasks.publish.dependencies, (std::vector<TaskId>{TaskId{11}}));
  ASSERT_TRUE(tasks.publish.publishes.has_value());
  EXPECT_EQ(tasks.publish.publishes->value, "gemm:B:packed");
  EXPECT_EQ(tasks.dormant_fallback.kind, TaskKind::kFallback);
  EXPECT_TRUE(tasks.dormant_fallback.dormant);
}

TEST(MaterializationRecipe, PackedPayloadSkipsPrepack) {
  PreparedRequirementDescriptor requirement{
      PreparedObjectRequirement{PreparedKey{"gemm:B:packed"}, "B"},
      {MaterializationRecipe{MaterializationRecipeKind::kReadPackedPayload, "B.packed",
                             "gemm-transB=0"}}};
  const MaterializationTaskDescriptors tasks =
      ExpandMaterializationRecipe(requirement, requirement.recipes.front(), TaskId{20});

  EXPECT_TRUE(tasks.prepack.dormant);
  EXPECT_EQ(tasks.publish.dependencies, (std::vector<TaskId>{TaskId{20}}));
  EXPECT_EQ(tasks.load.payload_id, "B.packed");
}

TEST(PreparedExecutionPlan, ConnectsSelectedRecipesAndExactPreparedKeyDependencies) {
  const RequiredPayloadManifest manifest = RequiredPayloadManifest::Freeze({
      PayloadManifestEntry{"embedding.packed", "prepared.bin", 0, 8},
      PayloadManifestEntry{"decoder.source", "weights.bin", 8, 16},
      PayloadManifestEntry{"embedding.source", "weights.bin", 0, 8,
                           PayloadResolution::kPreparedCacheReplaced},
  });
  PreparedRequirementDescriptor embedding{
      PreparedObjectRequirement{PreparedKey{"embedding"}, "embedding.source"},
      {MaterializationRecipe{MaterializationRecipeKind::kReadPackedPayload, "embedding.packed",
                             "embedding-layout"}}};
  PreparedRequirementDescriptor decoder{
      PreparedObjectRequirement{PreparedKey{"decoder.1"}, "decoder.source"},
      {MaterializationRecipe{MaterializationRecipeKind::kReadSourceAndPrepack, "decoder.source",
                             "decoder-layout"}}};
  TaskDescriptor first_token{TaskId{20}, TaskScope::kInvocation, TaskKind::kExecute,
                             ResourceClass::kCpu};
  first_token.prepared_requirements = {embedding.requirement.key};
  TaskDescriptor later{
      TaskId{21}, TaskScope::kInvocation, TaskKind::kExecute, ResourceClass::kCpu, {TaskId{20}}};
  later.prepared_requirements = {decoder.requirement.key};
  later.priority = TaskPriority::kPrefetch;

  PreparedExecutionPlan plan({first_token, later},
                             {
                                 PreparedMaterialization{embedding, embedding.recipes.front(),
                                                         TaskId{1}, TaskPriority::kCritical},
                                 PreparedMaterialization{decoder, decoder.recipes.front(),
                                                         TaskId{5}, TaskPriority::kPrefetch},
                             },
                             manifest);

  ASSERT_EQ(plan.tasks().size(), 10u);
  EXPECT_EQ(plan.tasks()[0].payload_id, "embedding.packed");
  EXPECT_TRUE(plan.tasks()[1].dormant);
  EXPECT_EQ(plan.tasks()[4].payload_id, "decoder.source");
  EXPECT_EQ(plan.tasks()[8].dependencies, (std::vector<TaskId>{TaskId{3}}));
  EXPECT_EQ(plan.tasks()[9].dependencies, (std::vector<TaskId>{TaskId{20}, TaskId{7}}));
}

TEST(PreparedExecutionPlan, TraceProvesFirstTokenOverlapsLaterPreparationWithinBudget) {
  const RequiredPayloadManifest manifest = RequiredPayloadManifest::Freeze({
      PayloadManifestEntry{"embedding", "weights.bin", 0, 4},
      PayloadManifestEntry{"decoder.1", "weights.bin", 4, 4},
  });
  PreparedRequirementDescriptor embedding{
      PreparedObjectRequirement{PreparedKey{"embedding.prepared"}, "embedding"},
      {MaterializationRecipe{MaterializationRecipeKind::kReadSourceAndPrepack, "embedding",
                             "embedding-layout"}}};
  PreparedRequirementDescriptor decoder{
      PreparedObjectRequirement{PreparedKey{"decoder.1.prepared"}, "decoder.1"},
      {MaterializationRecipe{MaterializationRecipeKind::kReadSourceAndPrepack, "decoder.1",
                             "decoder-layout"}}};
  TaskDescriptor first_token{TaskId{20}, TaskScope::kInvocation, TaskKind::kExecute,
                             ResourceClass::kCpu};
  first_token.prepared_requirements = {embedding.requirement.key};
  TaskDescriptor later{
      TaskId{21}, TaskScope::kInvocation, TaskKind::kExecute, ResourceClass::kCpu, {TaskId{20}}};
  later.prepared_requirements = {decoder.requirement.key};
  later.priority = TaskPriority::kPrefetch;
  PreparedExecutionPlan plan({first_token, later},
                             {
                                 PreparedMaterialization{embedding, embedding.recipes.front(),
                                                         TaskId{1}, TaskPriority::kCritical},
                                 PreparedMaterialization{decoder, decoder.recipes.front(),
                                                         TaskId{5}, TaskPriority::kPrefetch},
                             },
                             manifest);
  PreparedExecutionState state(
      1, 2, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(),
      PreparedSchedulerOptions{.io_workers = 2, .global_memory_budget = 8});

  const PreparedExecutionResult result =
      plan.RunSequential(state, [&](const TaskDescriptor &task, PreparedExecutionState &run_state) {
        if (task.kind == TaskKind::kReadPayload) {
          std::this_thread::sleep_for(std::chrono::milliseconds(task.id.value == 5 ? 30 : 2));
        } else if (task.kind == TaskKind::kPublish) {
          const PreparedObjectRequirement requirement{*task.publishes, task.payload_id};
          PreparedObjectRequest request = run_state.objects().Request(requirement);
          AllocationHandle allocation(&run_state.prepared_arena(),
                                      run_state.prepared_arena().Allocate(1));
          run_state.objects().Publish(request, std::move(allocation));
        }
      });

  const auto trace_for = [&](uint64_t task_id) {
    return std::find_if(result.trace.begin(), result.trace.end(),
                        [task_id](const auto &event) { return event.task_id.value == task_id; });
  };
  const auto later_read = trace_for(5);
  const auto first_execution = trace_for(20);
  const auto early_read = trace_for(1);
  ASSERT_NE(later_read, result.trace.end());
  ASSERT_NE(first_execution, result.trace.end());
  ASSERT_NE(early_read, result.trace.end());
  EXPECT_LT(first_execution->start_ns, later_read->end_ns);
  EXPECT_EQ(early_read->priority, TaskPriority::kCritical);
  EXPECT_EQ(later_read->priority, TaskPriority::kPrefetch);
  EXPECT_GT(result.peak_in_flight_bytes, 0u);
  EXPECT_LE(result.peak_in_flight_bytes, 8u);
}

TEST(PreparedTensorCache, MissPublishesAndPersistsReusableAtomicEntry) {
  const std::filesystem::path path = "prepared_tensor_cache_entry.bin";
  std::filesystem::remove(path);
  int source_reads = 0;
  int prepacks = 0;
  std::vector<uint8_t> published;
  PreparedTensorCache cache;
  const PreparedTensorLoadResult miss = cache.LoadOrPrepare(
      path, TestPreparedMetadata(),
      [&]() {
        ++source_reads;
        return std::vector<uint8_t>{1, 2, 3};
      },
      [&](const std::vector<uint8_t> &source) {
        ++prepacks;
        return std::vector<uint8_t>{source[2], source[1], source[0]};
      },
      [&](const std::vector<uint8_t> &prepared) { published = prepared; });

  EXPECT_FALSE(miss.cache_hit);
  EXPECT_EQ(miss.miss_reason, PreparedCacheMissReason::kNotFound);
  EXPECT_EQ(source_reads, 1);
  EXPECT_EQ(prepacks, 1);
  EXPECT_EQ(published, (std::vector<uint8_t>{3, 2, 1}));
  cache.WaitForBackgroundWrites();
  EXPECT_TRUE(cache.TakePersistenceDiagnostics().empty());
  EXPECT_TRUE(std::filesystem::exists(path));
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(path.parent_path().empty() ? "." : path.parent_path())) {
    EXPECT_EQ(entry.path().filename().string().find(path.filename().string() + ".tmp."),
              std::string::npos);
  }

  published.clear();
  const PreparedTensorLoadResult hit = cache.LoadOrPrepare(
      path, TestPreparedMetadata(),
      [&]() {
        ++source_reads;
        return std::vector<uint8_t>{9};
      },
      [&](const std::vector<uint8_t> &source) {
        ++prepacks;
        return source;
      },
      [&](const std::vector<uint8_t> &prepared) { published = prepared; });
  EXPECT_TRUE(hit.cache_hit);
  EXPECT_EQ(hit.miss_reason, PreparedCacheMissReason::kNone);
  EXPECT_EQ(source_reads, 1);
  EXPECT_EQ(prepacks, 1);
  EXPECT_EQ(published, (std::vector<uint8_t>{3, 2, 1}));
  std::filesystem::remove(path);
}

TEST(PreparedTensorCache, DiagnosesStaleIsaLayoutRuntimeAndFormatMisses) {
  const std::filesystem::path path = "prepared_tensor_cache_incompatible.bin";
  PreparedTensorMetadata cached = TestPreparedMetadata();
  cached.required_isa.Add(ONNX_LIGHT_NAMESPACE::core::platform::CpuFeature::kAvx2);

  const auto expect_miss = [&](PreparedTensorMetadata required,
                               PreparedCacheMissReason expected_reason) {
    std::filesystem::remove(path);
    PopulatePreparedCache(path, cached);
    PreparedTensorCache cache;
    const PreparedTensorLoadResult result = cache.LoadOrPrepare(
        path, required, []() { return std::vector<uint8_t>{4}; },
        [](const std::vector<uint8_t> &source) { return source; },
        [](const std::vector<uint8_t> &) {});
    EXPECT_FALSE(result.cache_hit);
    EXPECT_EQ(result.miss_reason, expected_reason);
    EXPECT_FALSE(result.diagnostic.empty());
    cache.WaitForBackgroundWrites();
  };

  PreparedTensorMetadata required = cached;
  required.source_digest = "new-source";
  expect_miss(required, PreparedCacheMissReason::kStaleSource);
  required = cached;
  required.required_isa = {};
  expect_miss(required, PreparedCacheMissReason::kWrongIsa);
  required = cached;
  required.kernel_layout = "gemm-transB=1";
  expect_miss(required, PreparedCacheMissReason::kWrongLayout);
  required = cached;
  required.runtime_version = "2";
  expect_miss(required, PreparedCacheMissReason::kWrongRuntime);
  required = cached;
  required.format_version = "2";
  expect_miss(required, PreparedCacheMissReason::kWrongFormat);
  std::filesystem::remove(path);
}

TEST(PreparedTensorCache, DiagnosesCorruptPayloadAsMiss) {
  const std::filesystem::path path = "prepared_tensor_cache_corrupt.bin";
  PopulatePreparedCache(path, TestPreparedMetadata());
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);

  PreparedTensorCache cache;
  const PreparedTensorLoadResult result = cache.LoadOrPrepare(
      path, TestPreparedMetadata(), []() { return std::vector<uint8_t>{7}; },
      [](const std::vector<uint8_t> &source) { return source; },
      [](const std::vector<uint8_t> &) {});
  EXPECT_FALSE(result.cache_hit);
  EXPECT_EQ(result.miss_reason, PreparedCacheMissReason::kCorrupt);
  EXPECT_NE(result.diagnostic.find("corrupt"), std::string::npos);
  cache.WaitForBackgroundWrites();

  {
    std::fstream entry(path, std::ios::binary | std::ios::in | std::ios::out);
    entry.seekp(-19, std::ios::end);
    const uint64_t impossible_size = std::numeric_limits<uint64_t>::max();
    entry.write(reinterpret_cast<const char *>(&impossible_size), sizeof(impossible_size));
  }
  const PreparedTensorLoadResult impossible_length = cache.LoadOrPrepare(
      path, TestPreparedMetadata(), []() { return std::vector<uint8_t>{8}; },
      [](const std::vector<uint8_t> &source) { return source; },
      [](const std::vector<uint8_t> &) {});
  EXPECT_FALSE(impossible_length.cache_hit);
  EXPECT_EQ(impossible_length.miss_reason, PreparedCacheMissReason::kCorrupt);
  EXPECT_NE(impossible_length.diagnostic.find("corrupt"), std::string::npos);
  cache.WaitForBackgroundWrites();
  std::filesystem::remove(path);
}

TEST(PreparedExecutionPlan, RejectsUnstableDependencies) {
  EXPECT_THROW(PreparedExecutionPlan({
                   TaskDescriptor{TaskId{2},
                                  TaskScope::kInvocation,
                                  TaskKind::kExecute,
                                  ResourceClass::kCpu,
                                  {TaskId{1}}},
               }),
               std::runtime_error);
}

TEST(PreparedExecutionPlan, RetriesFailedSessionGenerationAndKeepsInvocationsSeparate) {
  PreparedExecutionPlan plan({
      TaskDescriptor{TaskId{10}, TaskScope::kSession, TaskKind::kPrepare, ResourceClass::kCpu},
      TaskDescriptor{TaskId{20},
                     TaskScope::kInvocation,
                     TaskKind::kExecute,
                     ResourceClass::kCpu,
                     {TaskId{10}}},
  });
  PreparedExecutionState state;
  int session_executions = 0;
  int invocation_executions = 0;

  const PreparedExecutionResult failed =
      plan.RunSequential(state, [&](const TaskDescriptor &task, PreparedExecutionState &) {
        if (task.scope == TaskScope::kSession) {
          ++session_executions;
          throw std::runtime_error("weight unavailable");
        }
        ++invocation_executions;
      });
  ASSERT_EQ(failed.diagnostics.size(), 2u);
  EXPECT_EQ(failed.diagnostics[0].task_id.value, 10u);
  EXPECT_EQ(failed.diagnostics[0].status, TaskStatus::kFailed);
  EXPECT_EQ(failed.diagnostics[1].task_id.value, 20u);
  EXPECT_EQ(failed.diagnostics[1].status, TaskStatus::kSuppressed);
  ASSERT_TRUE(failed.diagnostics[1].caused_by.has_value());
  EXPECT_EQ(failed.diagnostics[1].caused_by->value, 10u);

  const auto successful_executor = [&](const TaskDescriptor &task, PreparedExecutionState &) {
    if (task.scope == TaskScope::kSession) {
      ++session_executions;
    } else {
      ++invocation_executions;
    }
  };
  const PreparedExecutionResult second = plan.RunSequential(state, successful_executor);
  const PreparedExecutionResult third = plan.RunSequential(state, successful_executor);

  EXPECT_NE(failed.invocation_id, second.invocation_id);
  EXPECT_NE(second.invocation_id, third.invocation_id);
  EXPECT_EQ(session_executions, 2);
  EXPECT_EQ(invocation_executions, 2);
  ASSERT_EQ(failed.session_generations.size(), 1u);
  ASSERT_EQ(second.session_generations.size(), 1u);
  ASSERT_EQ(third.session_generations.size(), 1u);
  EXPECT_EQ(failed.session_generations[0].second, 1u);
  EXPECT_EQ(second.session_generations[0].second, 2u);
  EXPECT_EQ(third.session_generations[0].second, 2u);
  EXPECT_EQ(second.diagnostics[1].status, TaskStatus::kSucceeded);
  EXPECT_EQ(third.diagnostics[1].status, TaskStatus::kSucceeded);
}

TEST(PreparedExecutionPlan, ParallelModeUsesSuppliedCpuExecutorAndMatchesSequentialResult) {
  PreparedExecutionPlan plan({
      TaskDescriptor{TaskId{1}, TaskScope::kSession, TaskKind::kReadPayload, ResourceClass::kIo},
      TaskDescriptor{
          TaskId{2}, TaskScope::kSession, TaskKind::kPrepare, ResourceClass::kCpu, {TaskId{1}}},
      TaskDescriptor{
          TaskId{3}, TaskScope::kInvocation, TaskKind::kExecute, ResourceClass::kCpu, {TaskId{2}}},
  });
  PreparedExecutionState sequential_state;
  PreparedExecutionState parallel_state;
  std::atomic<int> sequential_value{0};
  std::atomic<int> parallel_value{0};

  const PreparedExecutionResult sequential = plan.RunSequential(
      sequential_state, [&](const TaskDescriptor &task, PreparedExecutionState &) {
        sequential_value.fetch_add(static_cast<int>(task.id.value));
      });

  CpuExecutionPolicy policy;
  policy.num_threads = 2;
  const std::shared_ptr<CpuExecutor> cpu_executor = GlobalCpuExecutorRegistry().Acquire(policy);
  const size_t live_pools = GlobalCpuExecutorRegistry().live_pool_count();
  std::atomic<bool> used_supplied_executor{true};
  const PreparedExecutionResult parallel = plan.RunParallel(
      parallel_state,
      [&](const TaskDescriptor &task, PreparedExecutionState &) {
        if (task.resource == ResourceClass::kCpu) {
          used_supplied_executor =
              used_supplied_executor && CurrentCpuExecutor() == cpu_executor.get();
        }
        parallel_value.fetch_add(static_cast<int>(task.id.value));
      },
      *cpu_executor);

  EXPECT_EQ(parallel_value, sequential_value);
  EXPECT_TRUE(used_supplied_executor);
  EXPECT_EQ(GlobalCpuExecutorRegistry().live_pool_count(), live_pools);
  ASSERT_EQ(parallel.diagnostics.size(), sequential.diagnostics.size());
  for (size_t i = 0; i < parallel.diagnostics.size(); ++i) {
    EXPECT_EQ(parallel.diagnostics[i].status, sequential.diagnostics[i].status);
  }
}

TEST(PreparedExecutionPlan, InheritsCriticalPriorityAndReservesIoAdmission) {
  TaskDescriptor background{TaskId{1}, TaskScope::kSession, TaskKind::kPersist, ResourceClass::kIo};
  background.priority = TaskPriority::kBackground;
  TaskDescriptor required{TaskId{2}, TaskScope::kSession, TaskKind::kReadPayload,
                          ResourceClass::kIo};
  required.priority = TaskPriority::kBackground;
  PreparedExecutionPlan plan({
      background,
      required,
      TaskDescriptor{
          TaskId{3}, TaskScope::kInvocation, TaskKind::kExecute, ResourceClass::kCpu, {TaskId{2}}},
  });
  PreparedExecutionState state(
      1, 1, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(),
      PreparedSchedulerOptions{.io_workers = 1, .reserved_critical_memory = 1});
  std::mutex order_mutex;
  std::vector<uint64_t> order;

  plan.RunSequential(state, [&](const TaskDescriptor &task, PreparedExecutionState &) {
    std::lock_guard<std::mutex> lock(order_mutex);
    order.push_back(task.id.value);
    if (task.id.value == 2) {
      EXPECT_EQ(task.priority, TaskPriority::kCritical);
    }
  });

  ASSERT_EQ(order.size(), 3u);
  EXPECT_EQ(order[0], 2u);
  EXPECT_EQ(order[1], 3u);
  EXPECT_EQ(order[2], 1u);
}

TEST(PreparedExecutionPlan, EnforcesGlobalAndIoMemoryBudgets) {
  TaskDescriptor first{TaskId{1}, TaskScope::kSession, TaskKind::kReadPayload, ResourceClass::kIo};
  first.estimated_input_bytes = 8;
  TaskDescriptor second{TaskId{2}, TaskScope::kSession, TaskKind::kReadPayload, ResourceClass::kIo};
  second.estimated_input_bytes = 8;
  PreparedExecutionPlan plan({first, second});
  PreparedExecutionState state(
      1, 1, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(),
      PreparedSchedulerOptions{.io_workers = 2, .global_memory_budget = 12, .io_memory_budget = 8});
  std::atomic<int> active{0};
  std::atomic<int> maximum_active{0};

  const PreparedExecutionResult result =
      plan.RunSequential(state, [&](const TaskDescriptor &, PreparedExecutionState &) {
        const int current = ++active;
        maximum_active = std::max(maximum_active.load(), current);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        --active;
      });

  EXPECT_EQ(maximum_active, 1);
  EXPECT_LE(result.peak_in_flight_bytes, 8u);
}

TEST(PreparedExecutionPlan, EnforcesPreparedAndExecutionMemoryBudgets) {
  TaskDescriptor prepare{TaskId{1}, TaskScope::kSession, TaskKind::kPrepare, ResourceClass::kCpu};
  prepare.estimated_output_bytes = 9;
  PreparedExecutionPlan preparation_plan({prepare});
  PreparedExecutionState preparation_state(1, 1, std::numeric_limits<size_t>::max(),
                                           std::numeric_limits<size_t>::max(),
                                           PreparedSchedulerOptions{.prepared_memory_budget = 8});
  EXPECT_THROW(preparation_plan.RunSequential(
                   preparation_state, [](const TaskDescriptor &, PreparedExecutionState &) {}),
               std::runtime_error);

  TaskDescriptor execute{TaskId{1}, TaskScope::kInvocation, TaskKind::kExecute,
                         ResourceClass::kCpu};
  execute.peak_temporary_bytes = 9;
  PreparedExecutionPlan execution_plan({execute});
  PreparedExecutionState execution_state(1, 1, std::numeric_limits<size_t>::max(),
                                         std::numeric_limits<size_t>::max(),
                                         PreparedSchedulerOptions{.execution_memory_budget = 8});
  EXPECT_THROW(execution_plan.RunSequential(
                   execution_state, [](const TaskDescriptor &, PreparedExecutionState &) {}),
               std::runtime_error);
}

TEST(PreparedExecutionPlan, FullyPreparedRunUsesEpochHotPathWithoutEnqueues) {
  const PreparedObjectRequirement requirement{PreparedKey{"hot"}, "source"};
  TaskDescriptor publish{TaskId{1}, TaskScope::kSession, TaskKind::kPublish,
                         ResourceClass::kInline};
  publish.publishes = requirement.key;
  TaskDescriptor first_action{
      TaskId{2}, TaskScope::kInvocation, TaskKind::kExecute, ResourceClass::kCpu, {TaskId{1}}};
  first_action.prepared_requirements = {requirement.key};
  TaskDescriptor second_action{
      TaskId{3}, TaskScope::kInvocation, TaskKind::kExecute, ResourceClass::kCpu, {TaskId{2}}};
  second_action.prepared_requirements = {requirement.key};
  PreparedExecutionPlan plan({publish, first_action, second_action});
  PreparedExecutionState state;
  int invocation_executions = 0;
  const auto executor = [&](const TaskDescriptor &task, PreparedExecutionState &run_state) {
    if (task.scope == TaskScope::kSession) {
      PreparedObjectRequest request = run_state.objects().Request(requirement);
      AllocationHandle allocation(&run_state.prepared_arena(),
                                  run_state.prepared_arena().Allocate(1));
      run_state.objects().Publish(request, std::move(allocation));
    } else {
      ++invocation_executions;
    }
  };

  const PreparedExecutionResult cold = plan.RunSequential(state, executor);
  const PreparedExecutionResult hot = plan.RunSequential(state, executor);

  EXPECT_FALSE(cold.used_hot_path);
  EXPECT_EQ(cold.enqueued_tasks, 3u);
  EXPECT_TRUE(hot.used_hot_path);
  EXPECT_EQ(hot.enqueued_tasks, 0u);
  EXPECT_EQ(invocation_executions, 4);

  bool remained_pinned = false;
  const PreparedExecutionResult evicting =
      plan.RunSequential(state, [&](const TaskDescriptor &task, PreparedExecutionState &run_state) {
        if (task.scope == TaskScope::kInvocation && task.id.value == 2) {
          EXPECT_TRUE(run_state.objects().Evict(requirement.key));
          remained_pinned = run_state.prepared_arena().allocated_count() == 1;
        }
      });
  EXPECT_TRUE(evicting.used_hot_path);
  EXPECT_TRUE(remained_pinned);
  EXPECT_EQ(state.prepared_arena().allocated_count(), 0u);

  const PreparedExecutionResult reloaded = plan.RunSequential(state, executor);
  EXPECT_FALSE(reloaded.used_hot_path);
  EXPECT_EQ(reloaded.enqueued_tasks, 3u);
}
