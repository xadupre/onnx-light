#include "onnx_io_policy.h"
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>

using ONNX_LIGHT_NAMESPACE::utils::IOPolicy;
using ONNX_LIGHT_NAMESPACE::utils::IOStorageKind;

namespace {

TEST(onnx_io_policy, MmapAlwaysResolvesToZeroWorkers) {
  // Regardless of the requested thread count, a memory-mapped read never needs a worker pool:
  // bytes are resolved through lazy page faults, not by copying through the thread pool.
  IOPolicy auto_policy =
      ONNX_LIGHT_NAMESPACE::utils::ResolveIOPolicy(IOStorageKind::kMmap, 1 << 20, -1, 0);
  EXPECT_EQ(auto_policy.workers, 0);

  IOPolicy explicit_policy =
      ONNX_LIGHT_NAMESPACE::utils::ResolveIOPolicy(IOStorageKind::kMmap, 1 << 20, 8, 0);
  EXPECT_EQ(explicit_policy.workers, 0);
}

TEST(onnx_io_policy, ExplicitRequestIsHonoredForNonMmapKinds) {
  for (IOStorageKind kind : {IOStorageKind::kWarmPageCache, IOStorageKind::kBufferedReads,
                             IOStorageKind::kColdStorage}) {
    IOPolicy serial =
        ONNX_LIGHT_NAMESPACE::utils::ResolveIOPolicy(kind, 1 << 20, /*requested_num_threads=*/1, 0);
    EXPECT_EQ(serial.workers, 1) << "kind=" << static_cast<int>(kind);

    IOPolicy explicit_n =
        ONNX_LIGHT_NAMESPACE::utils::ResolveIOPolicy(kind, 1 << 20, /*requested_num_threads=*/5, 0);
    EXPECT_EQ(explicit_n.workers, 5) << "kind=" << static_cast<int>(kind);
  }
}

TEST(onnx_io_policy, AutoWarmPageCacheUsesMoreWorkersThanCold) {
  if (std::thread::hardware_concurrency() < 2) {
    GTEST_SKIP() << "Needs at least 2 logical processors to distinguish per-kind ceilings.";
  }
  // Warm (page-cache resident) reads are memory-bandwidth bound, so the calibrated defaults
  // should offer more worker threads than cold-storage reads for the same payload size.
  const int64_t total_bytes = 256 << 20; // 256 MiB, large enough not to be capped by total_bytes.
  IOPolicy warm = ONNX_LIGHT_NAMESPACE::utils::ResolveIOPolicy(IOStorageKind::kWarmPageCache,
                                                               total_bytes, -1, 0);
  IOPolicy cold =
      ONNX_LIGHT_NAMESPACE::utils::ResolveIOPolicy(IOStorageKind::kColdStorage, total_bytes, -1, 0);
  EXPECT_GT(warm.workers, cold.workers);
  EXPECT_GT(cold.min_block_size, warm.min_block_size);
}

TEST(onnx_io_policy, AutoWorkerCountNeverExceedsUsefulBlockCount) {
  // A handful of bytes should never justify spinning up a whole worker pool: the automatic
  // choice must be capped by total_bytes / min_block_size.
  IOPolicy policy = ONNX_LIGHT_NAMESPACE::utils::ResolveIOPolicy(IOStorageKind::kColdStorage,
                                                                 /*total_bytes=*/1, -1, 0);
  EXPECT_LE(policy.workers, 1);
}

TEST(onnx_io_policy, ExplicitMinBlockSizeOverridesDefault) {
  IOPolicy policy = ONNX_LIGHT_NAMESPACE::utils::ResolveIOPolicy(
      IOStorageKind::kColdStorage, 1 << 20, -1, /*requested_min_block_size=*/12345);
  EXPECT_EQ(policy.min_block_size, 12345);
}

TEST(onnx_io_policy, DetectIOStorageKindFallsBackForMissingFile) {
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::utils::DetectIOStorageKind("/no/such/file/here.bin"),
            IOStorageKind::kBufferedReads);
}

TEST(onnx_io_policy, DetectIOStorageKindHandlesRealFileWithoutCrashing) {
  std::string path = "test_io_policy_detect_real_file.bin";
  {
    std::ofstream f(path, std::ios::binary);
    std::vector<char> buf(4096, 'a');
    f.write(buf.data(), static_cast<std::streamsize>(buf.size()));
  }
  // Just after writing, the file is typically resident in the page cache, but the exact
  // classification is platform/OS dependent; only require that it returns a valid enumerator
  // without crashing.
  IOStorageKind kind = ONNX_LIGHT_NAMESPACE::utils::DetectIOStorageKind(path);
  EXPECT_TRUE(kind == IOStorageKind::kWarmPageCache || kind == IOStorageKind::kColdStorage ||
              kind == IOStorageKind::kBufferedReads);
  std::remove(path.c_str());
}

} // namespace
