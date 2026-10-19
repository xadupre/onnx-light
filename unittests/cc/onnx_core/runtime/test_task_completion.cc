// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/task_completion.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>

using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

TEST(TaskCompletion, WaitsAndSharesSuccess) {
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

TEST(TaskCompletion, PreservesFailureDiagnosticAndException) {
  TaskCompletion completion(TaskId{5});
  completion.Fail(std::make_exception_ptr(std::runtime_error("operation failed")),
                  "failed operation");

  EXPECT_THROW(completion.Wait(), std::runtime_error);
  TaskDiagnostic diagnostic = completion.diagnostic();
  EXPECT_EQ(diagnostic.task_id.value, 5);
  EXPECT_EQ(diagnostic.status, TaskStatus::kFailed);
  EXPECT_EQ(diagnostic.message, "failed operation");
}

TEST(TaskCompletion, RecordsSuppressedDependency) {
  TaskCompletion completion(TaskId{6});
  completion.Suppress(TaskId{5}, "dependency failed");

  completion.Wait();
  TaskDiagnostic diagnostic = completion.diagnostic();
  EXPECT_EQ(diagnostic.status, TaskStatus::kSuppressed);
  ASSERT_TRUE(diagnostic.caused_by.has_value());
  EXPECT_EQ(diagnostic.caused_by->value, 5);
}
