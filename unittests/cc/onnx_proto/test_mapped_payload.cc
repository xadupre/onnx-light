// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the ORT-independent MappedPayload ownership contract
// (onnx_light/onnx_proto/mapped_payload.h). These exercise the acceptance
// criteria from the "expose the onnxruntime payload ownership contract"
// roadmap item: owner release, alignment, file replacement, truncation,
// concurrent views, and path confinement.
#include "onnx_mapped_payload.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

class MappedPayloadTest : public ::testing::Test {
protected:
  void SetUp() override {
    base_dir_ = std::filesystem::temp_directory_path() /
                ("onnx_light_mapped_payload_" +
                 std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                 std::to_string(reinterpret_cast<uintptr_t>(this)));
    std::filesystem::create_directories(base_dir_);
  }

  void TearDown() override { std::filesystem::remove_all(base_dir_); }

  std::string WriteFile(const std::string &name, const std::string &content) const {
    const std::filesystem::path path = base_dir_ / name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    out.close();
    return path.string();
  }

  std::filesystem::path base_dir_;
};

std::string MakeBytes(size_t n, char fill_start = 0) {
  std::string result(n, '\0');
  for (size_t i = 0; i < n; ++i) {
    result[i] = static_cast<char>(fill_start + static_cast<char>(i));
  }
  return result;
}

} // namespace

TEST_F(MappedPayloadTest, BorrowReturnsExpectedBytesAndAlignment) {
  const std::string content = MakeBytes(64, 1);
  WriteFile("weights.bin", content);

  MappedPayloadSource source(base_dir_.string());
  MappedPayload payload = source.Borrow("weights.bin", 0, content.size());
  ASSERT_NE(payload.data, nullptr);
  EXPECT_EQ(payload.size, content.size());
  ASSERT_NE(payload.owner, nullptr);
  // The base of a memory-mapped file is page-aligned, so offset 0 has at least
  // the page size worth of alignment (>= 4096, i.e. much more than 8 bytes).
  EXPECT_GE(payload.alignment, size_t{8});
  EXPECT_EQ(std::memcmp(payload.data, content.data(), content.size()), 0);
}

TEST_F(MappedPayloadTest, BorrowNonZeroOffsetReportsReducedAlignment) {
  const std::string content = MakeBytes(256, 5);
  WriteFile("weights.bin", content);

  MappedPayloadSource source(base_dir_.string());
  // Offset 4 is not page-aligned; alignment should reflect that (a power of two
  // no greater than 4, since 4 is the largest power of two dividing 4).
  MappedPayload payload = source.Borrow("weights.bin", 4, 16);
  EXPECT_LE(payload.alignment, size_t{4});
  EXPECT_EQ(std::memcmp(payload.data, content.data() + 4, 16), 0);
}

TEST_F(MappedPayloadTest, OwnerKeepsMappingAliveAfterSourceReleasesCache) {
  const std::string content = MakeBytes(32, 9);
  WriteFile("weights.bin", content);

  MappedPayloadSource source(base_dir_.string());
  MappedPayload payload = source.Borrow("weights.bin", 0, content.size());
  std::shared_ptr<void> kept_owner = payload.owner;
  const void *kept_data = payload.data;

  // Dropping the source's cache (and even the source itself) must not
  // invalidate a payload whose owner is still held by the caller.
  source.ReleaseCachedMappings();
  EXPECT_EQ(std::memcmp(kept_data, content.data(), content.size()), 0);

  // Once every copy of owner is released, the mapping unmaps; there is no
  // portable way to observe that fact directly from this process without
  // reintroducing undefined behavior, so this test documents the contract
  // (owner keeps `data` valid) rather than asserting on unmap timing.
  kept_owner.reset();
  EXPECT_EQ(kept_owner, nullptr);
}

TEST_F(MappedPayloadTest, OutOfRangeRequestThrows) {
  const std::string content = MakeBytes(16, 0);
  WriteFile("weights.bin", content);

  MappedPayloadSource source(base_dir_.string());
  EXPECT_THROW(source.Borrow("weights.bin", 8, 16), std::exception);
  EXPECT_THROW(source.Borrow("weights.bin", -1, 4), std::exception);
  EXPECT_THROW(source.Borrow("weights.bin", 0, -1), std::exception);
}

TEST_F(MappedPayloadTest, FileReplacementProducesNewGenerationAndBytes) {
  const std::string original = MakeBytes(32, 1);
  WriteFile("weights.bin", original);

  MappedPayloadSource source(base_dir_.string());
  MappedPayload first = source.Borrow("weights.bin", 0, original.size());
  const PayloadIdentity first_identity = first.identity;

  // Simulate an atomic file replacement (a new inode swapped in via rename,
  // as a model-publishing tool would do) rather than an in-place overwrite:
  // the old mapping must keep observing the old inode's bytes.
  const std::string replacement = MakeBytes(32, 100);
  const std::string replacement_path = WriteFile("weights.bin.new", replacement);
  std::filesystem::rename(replacement_path, base_dir_ / "weights.bin");
  source.ReleaseCachedMappings();

  MappedPayload second = source.Borrow("weights.bin", 0, replacement.size());
  EXPECT_NE(second.identity.generation, first_identity.generation);
  EXPECT_EQ(std::memcmp(second.data, replacement.data(), replacement.size()), 0);
  // The first payload's bytes remain those of the original file because its
  // owner still pins the original mapping (old inode).
  EXPECT_EQ(std::memcmp(first.data, original.data(), original.size()), 0);
}

TEST_F(MappedPayloadTest, TruncatedRangeIsRejectedByBorrow) {
  const std::string content = MakeBytes(16, 0);
  WriteFile("weights.bin", content);

  MappedPayloadSource source(base_dir_.string());
  // Shrink the file after the source has never mapped it yet.
  WriteFile("weights.bin", MakeBytes(8, 0));
  EXPECT_THROW(source.Borrow("weights.bin", 0, 16), std::exception);
}

TEST_F(MappedPayloadTest, ConcurrentBorrowsOfSameFileShareOwnerAndSucceed) {
  const std::string content = MakeBytes(4096, 3);
  WriteFile("weights.bin", content);

  MappedPayloadSource source(base_dir_.string());
  constexpr int kThreads = 8;
  std::vector<std::thread> threads;
  std::vector<MappedPayload> results(kThreads);
  std::atomic<int> failures{0};

  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i]() {
      try {
        results[i] = source.Borrow("weights.bin", 0, content.size());
      } catch (...) {
        failures.fetch_add(1);
      }
    });
  }
  for (auto &t : threads) {
    t.join();
  }

  EXPECT_EQ(failures.load(), 0);
  for (const auto &payload : results) {
    ASSERT_NE(payload.data, nullptr);
    EXPECT_EQ(std::memcmp(payload.data, content.data(), content.size()), 0);
  }
  // All concurrent borrows of the same file share the same underlying owner.
  for (int i = 1; i < kThreads; ++i) {
    EXPECT_EQ(results[i].owner, results[0].owner);
    EXPECT_EQ(results[i].identity.generation, results[0].identity.generation);
  }
}

TEST_F(MappedPayloadTest, PathOutsideBaseDirIsRejected) {
  MappedPayloadSource source(base_dir_.string());
  const std::filesystem::path outside =
      std::filesystem::temp_directory_path() / "onnx_light_mapped_payload_outside.bin";
  {
    std::ofstream out(outside, std::ios::binary | std::ios::trunc);
    out << "outside";
  }
  EXPECT_THROW(source.Borrow(outside.string(), 0, 4), std::exception);
  EXPECT_THROW(source.Borrow("../onnx_light_mapped_payload_outside.bin", 0, 4), std::exception);
  std::filesystem::remove(outside);
}

TEST_F(MappedPayloadTest, SymlinkSourceIsRejected) {
  const std::string content = MakeBytes(16, 0);
  const std::string real_path = WriteFile("real_weights.bin", content);
  const std::filesystem::path link_path = base_dir_ / "linked_weights.bin";
  std::error_code ec;
  std::filesystem::create_symlink(real_path, link_path, ec);
  if (ec) {
    GTEST_SKIP() << "Symlinks are not supported in this environment: " << ec.message();
  }

  MappedPayloadSource source(base_dir_.string());
  EXPECT_THROW(source.Borrow("linked_weights.bin", 0, 4), std::exception);
}

TEST_F(MappedPayloadTest, FinalDestinationReadDescriptorReadsDirectlyIntoCallerBuffer) {
  const std::string content = MakeBytes(48, 7);
  WriteFile("weights.bin", content);

  MappedPayloadSource source(base_dir_.string());
  FinalDestinationReadDescriptor descriptor =
      source.DescribeFinalDestinationRead("weights.bin", 8, 16);
  EXPECT_EQ(descriptor.offset(), 8);
  EXPECT_EQ(descriptor.size(), 16);

  std::vector<char> destination(16, '\0');
  descriptor.ReadInto(destination.data());
  EXPECT_EQ(std::memcmp(destination.data(), content.data() + 8, 16), 0);
}

TEST_F(MappedPayloadTest, FinalDestinationReadDescriptorRejectsTruncationAtReadTime) {
  const std::string content = MakeBytes(64, 2);
  WriteFile("weights.bin", content);

  MappedPayloadSource source(base_dir_.string());
  FinalDestinationReadDescriptor descriptor =
      source.DescribeFinalDestinationRead("weights.bin", 0, 64);

  // Truncate the file after validation but before ReadInto().
  WriteFile("weights.bin", MakeBytes(8, 2));

  std::vector<char> destination(64, '\0');
  EXPECT_THROW(descriptor.ReadInto(destination.data()), std::exception);
}

TEST_F(MappedPayloadTest, FinalDestinationReadDescriptorOutOfRangeThrows) {
  const std::string content = MakeBytes(16, 0);
  WriteFile("weights.bin", content);

  MappedPayloadSource source(base_dir_.string());
  EXPECT_THROW(source.DescribeFinalDestinationRead("weights.bin", 8, 16), std::exception);
}

TEST_F(MappedPayloadTest, PayloadIdentityEqualityReflectsRangeAndGeneration) {
  const std::string content = MakeBytes(32, 0);
  WriteFile("weights.bin", content);

  MappedPayloadSource source(base_dir_.string());
  MappedPayload first = source.Borrow("weights.bin", 0, 16);
  MappedPayload second = source.Borrow("weights.bin", 0, 16);
  EXPECT_EQ(first.identity, second.identity);

  MappedPayload different_range = source.Borrow("weights.bin", 16, 16);
  EXPECT_NE(first.identity, different_range.identity);
}
