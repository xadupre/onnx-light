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
#include "stream.h"
#include "windows_junction.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#endif

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

void ReplaceDirectoryWithLink(const std::filesystem::path &directory,
                              const std::filesystem::path &target) {
  std::filesystem::rename(directory, directory.string() + ".original");
#if defined(_WIN32)
  // Junctions do not require the symlink privilege on Windows.
  std::filesystem::create_directory(directory);
  HANDLE handle = CreateFileW(directory.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                              FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  ASSERT_NE(handle, INVALID_HANDLE_VALUE);
  struct JunctionData {
    DWORD tag;
    WORD data_length;
    WORD reserved;
    WORD substitute_offset;
    WORD substitute_length;
    WORD print_offset;
    WORD print_length;
    WCHAR path[2048];
  } data{};
  const std::wstring destination = std::filesystem::absolute(target).wstring();
  const std::wstring substitute = L"\\??\\" + destination;
  const size_t total_chars = substitute.size() + 1 + destination.size() + 1;
  if (total_chars > std::size(data.path)) {
    CloseHandle(handle);
    FAIL() << "Junction target exceeds the test buffer.";
  }
  data.tag = IO_REPARSE_TAG_MOUNT_POINT;
  data.substitute_length = static_cast<WORD>(substitute.size() * sizeof(WCHAR));
  data.print_offset = static_cast<WORD>((substitute.size() + 1) * sizeof(WCHAR));
  data.print_length = static_cast<WORD>(destination.size() * sizeof(WCHAR));
  std::memcpy(data.path, substitute.c_str(), (substitute.size() + 1) * sizeof(WCHAR));
  std::memcpy(reinterpret_cast<char *>(data.path) + data.print_offset, destination.c_str(),
              (destination.size() + 1) * sizeof(WCHAR));
  data.data_length = static_cast<WORD>(8 + total_chars * sizeof(WCHAR));
  DWORD returned = 0;
  const BOOL success = DeviceIoControl(handle, FSCTL_SET_REPARSE_POINT, &data, data.data_length + 8,
                                       nullptr, 0, &returned, nullptr);
  const DWORD error = GetLastError();
  CloseHandle(handle);
  ASSERT_TRUE(success) << "Cannot create test junction: " << error;
#else
  std::filesystem::create_directory_symlink(target, directory);
#endif
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

TEST_F(MappedPayloadTest, TrustedBaseAliasesPreserveMappedAndDirectReadsAndRejectOutsideChildren) {
  const auto base = base_dir_ / "model";
  const auto alias = base_dir_ / "alias";
  std::filesystem::create_directory(base);
  const auto content = MakeBytes(32, 3);
  WriteFile("model/weights.bin", content);
#ifdef _WIN32
  const auto error = onnx_light_test::CreateWindowsJunction(alias, base);
  ASSERT_FALSE(error) << error.message();
#else
  std::filesystem::create_directory_symlink(base, alias);
#endif
  for (const auto &source_base : {base, alias}) {
    MappedPayloadSource source(source_base.string());
    const auto relative = source.Borrow("weights.bin", 0, content.size());
    for (const auto &file_base : {base, alias}) {
      const auto path = (file_base / "weights.bin").string();
      const auto mapped = source.Borrow(path, 0, content.size());
      EXPECT_EQ(mapped.owner, relative.owner);
      EXPECT_EQ(mapped.identity.source_path, relative.identity.source_path);
      EXPECT_EQ(std::memcmp(mapped.data, content.data(), content.size()), 0);
      std::vector<uint8_t> bytes(content.size());
      source.DescribeFinalDestinationRead(path, 0, content.size()).ReadInto(bytes.data());
      EXPECT_EQ(std::memcmp(bytes.data(), content.data(), content.size()), 0);
    }
  }
#ifdef _WIN32
  std::filesystem::create_directory(base_dir_ / "outside");
  WriteFile("outside/weights.bin", content);
  const auto child_error =
      onnx_light_test::CreateWindowsJunction(base / "linked", base_dir_ / "outside");
  ASSERT_FALSE(child_error) << child_error.message();
#else
  std::filesystem::create_directory(base_dir_ / "outside");
  WriteFile("outside/weights.bin", content);
  std::filesystem::create_directory_symlink(base_dir_ / "outside", base / "linked");
#endif
  MappedPayloadSource source(alias.string());
  EXPECT_ANY_THROW(source.Borrow((base / "linked" / "weights.bin").string(), 0, 4));
  EXPECT_ANY_THROW(source.DescribeFinalDestinationRead("linked/weights.bin", 0, 4));
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
  EXPECT_THROW(source.Borrow("weights.bin", int64_t{-1}, 4), std::exception);
  EXPECT_THROW(source.Borrow("weights.bin", 0, int64_t{-1}), std::exception);
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

TEST_F(MappedPayloadTest, ConfinedNestedPathsRemainSupported) {
  std::filesystem::create_directories(base_dir_ / "first" / "second");
  const std::string content = MakeBytes(32, 7);
  WriteFile("first/second/weights.bin", content);
  MappedPayloadSource source(base_dir_.string());
  const auto payload = source.Borrow("first/second/weights.bin", 4, 16);
  EXPECT_EQ(std::memcmp(payload.data, content.data() + 4, 16), 0);
  const auto descriptor = source.DescribeFinalDestinationRead("first/second/weights.bin", 4, 16);
  std::vector<char> destination(16);
  descriptor.ReadInto(destination.data());
  EXPECT_EQ(std::memcmp(destination.data(), content.data() + 4, 16), 0);
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

TEST_F(MappedPayloadTest, FinalDestinationReadRetainsFileAcrossDirectoryReplacement) {
  std::filesystem::create_directories(base_dir_ / "model" / "child");
  std::filesystem::create_directory(base_dir_ / "outside");
  const std::string original = MakeBytes(32, 3);
  WriteFile("model/child/weights.bin", original);
  WriteFile("outside/weights.bin", MakeBytes(32, 99));
  MappedPayloadSource source((base_dir_ / "model").string());
  auto descriptor = source.DescribeFinalDestinationRead("child/weights.bin", 4, 16);

  ASSERT_NO_FATAL_FAILURE(
      ReplaceDirectoryWithLink(base_dir_ / "model" / "child", base_dir_ / "outside"));
  std::vector<char> destination(16);
  descriptor.ReadInto(destination.data());
  EXPECT_EQ(std::memcmp(destination.data(), original.data() + 4, 16), 0);
  EXPECT_THROW(source.DescribeFinalDestinationRead("child/weights.bin", 4, 16), std::exception);
  EXPECT_THROW(source.Borrow("child/weights.bin", 4, 16), std::exception);
}

TEST_F(MappedPayloadTest, MappingUsesOpenedFileAfterDirectoryReplacement) {
  std::filesystem::create_directories(base_dir_ / "model" / "child");
  std::filesystem::create_directory(base_dir_ / "outside");
  const std::string original = MakeBytes(32, 3);
  WriteFile("model/child/weights.bin", original);
  WriteFile("outside/weights.bin", MakeBytes(32, 99));
  const auto file =
      utils::OpenExternalDataFile((base_dir_ / "model").string(), "child/weights.bin");

  ASSERT_NO_FATAL_FAILURE(
      ReplaceDirectoryWithLink(base_dir_ / "model" / "child", base_dir_ / "outside"));
  const auto mapping = utils::MapExternalDataFile(*file);
  ASSERT_NE(mapping, nullptr);
  EXPECT_EQ(std::memcmp(mapping.get(), original.data(), original.size()), 0);
}

TEST_F(MappedPayloadTest, ParallelStreamReadsRetainFilesAcrossDirectoryReplacement) {
  const auto model = base_dir_ / "model";
  std::filesystem::create_directories(model / "child");
  std::filesystem::create_directory(base_dir_ / "outside");
  WriteFile("model/model.onnx", "model");
  WriteFile("model/child/weights.bin", "original");
  WriteFile("model/child/extra.bin", "extra!!!");
  WriteFile("outside/weights.bin", "sentinel");
  WriteFile("outside/extra.bin", "sentinel");
  utils::TwoFilesStream stream((model / "model.onnx").string(),
                               (model / "child" / "weights.bin").string(), true);
  stream.set_active_weights_location("child/extra.bin");
  stream.set_active_weights_location("child/weights.bin");
  ASSERT_NO_FATAL_FAILURE(ReplaceDirectoryWithLink(model / "child", base_dir_ / "outside"));

  uint8_t destination[8]{};
  stream.read_bytes_from_weights_stream(8, destination, 0);
  EXPECT_EQ(std::string(reinterpret_cast<char *>(destination), 8), "original");
  for (const auto &location : {"child/weights.bin", "child/extra.bin"}) {
    stream.set_active_weights_location(location);
    stream.read_bytes_from_weights_stream(0, destination, 0);
    stream.StartThreadPool(1);
    utils::DelayedBlock block{8, destination, 0, 1};
    stream.ReadDelayedBlock(block);
    stream.WaitForDelayedBlock();
    const std::string expected =
        location == std::string("child/weights.bin") ? "original" : "extra!!!";
    EXPECT_EQ(std::string(reinterpret_cast<char *>(destination), 8), expected);
    std::shared_ptr<void> owner;
    const auto *mapped = stream.borrow_weights_bytes(location, 0, 8, 1, owner);
    ASSERT_NE(mapped, nullptr);
    EXPECT_EQ(std::string(reinterpret_cast<const char *>(mapped), 8), expected);
  }
}

TEST_F(MappedPayloadTest, ParallelStreamWritesRetainFilesAcrossDirectoryReplacement) {
  const auto model = base_dir_ / "model";
  std::filesystem::create_directories(model / "child");
  std::filesystem::create_directory(base_dir_ / "outside");
  WriteFile("outside/weights.bin", "sentinel");
  WriteFile("outside/extra.bin", "sentinel");
  {
    utils::TwoFilesWriteStream stream((model / "model.onnx").string(),
                                      (model / "child" / "weights.bin").string());
    stream.pre_allocate_weights(12);
    stream.pre_allocate_weights("child/extra.bin", 12);
    ASSERT_NO_FATAL_FAILURE(ReplaceDirectoryWithLink(model / "child", base_dir_ / "outside"));
    stream.StartWriteThreadPool(2);
    stream.SetMinParallelBlockSize(8);
    for (const auto &location : {"child/weights.bin", "child/extra.bin"}) {
      stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>("original"), 8,
                                              location);
      stream.write_raw_bytes_in_second_stream(reinterpret_cast<const uint8_t *>("tail"), 4,
                                              location);
    }
    stream.WaitForWriteCompletion();
  }
  for (const auto &filename : {"weights.bin", "extra.bin"}) {
    std::ifstream pinned(model / "child.original" / filename, std::ios::binary);
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(pinned), {}), "originaltail");
    std::ifstream outside(base_dir_ / "outside" / filename, std::ios::binary);
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(outside), {}), "sentinel");
  }
}

TEST_F(MappedPayloadTest, FinalDestinationReadPinsFileAtConstruction) {
  const std::string original = MakeBytes(32, 3);
  const std::string path = WriteFile("weights.bin", original);
  FinalDestinationReadDescriptor descriptor(path, 4, 16, PayloadIdentity{});
  std::filesystem::rename(base_dir_ / "weights.bin", base_dir_ / "weights.original");
  WriteFile("weights.bin", MakeBytes(32, 99));
  std::vector<char> destination(16);
  descriptor.ReadInto(destination.data());
  EXPECT_EQ(std::memcmp(destination.data(), original.data() + 4, 16), 0);
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
