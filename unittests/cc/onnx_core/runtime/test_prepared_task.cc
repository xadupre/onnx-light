// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/prepared_task.h"
#include "onnx_core/compute/resolved_model_fixture.h"
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
