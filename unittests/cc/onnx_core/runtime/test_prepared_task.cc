// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/prepared_task.h"
#include "onnx_core/compute/prepared_tensor_cache.h"
#include "onnx_core/compute/resolved_model_fixture.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
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
