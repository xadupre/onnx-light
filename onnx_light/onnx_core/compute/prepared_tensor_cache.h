// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/platform/cpu_descriptor.h"
#include "onnx_light_helpers.h"

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

enum class PreparedCacheMissReason {
  kNone,
  kNotFound,
  kCorrupt,
  kStaleSource,
  kWrongArchitecture,
  kWrongIsa,
  kWrongRuntime,
  kWrongLayout,
  kWrongFormat,
};

struct PreparedTensorMetadata {
  std::string source_digest;
  std::string architecture;
  platform::CpuFeatureSet required_isa;
  std::string runtime;
  std::string runtime_version;
  std::string kernel_layout;
  std::string format_version;
};

struct PreparedTensorLoadResult {
  bool cache_hit = false;
  PreparedCacheMissReason miss_reason = PreparedCacheMissReason::kNone;
  std::string diagnostic;
};

using PreparedSourceLoader = std::function<std::vector<uint8_t>()>;
using PreparedTensorPrepacker = std::function<std::vector<uint8_t>(const std::vector<uint8_t> &)>;
using PreparedTensorPublisher = std::function<void(const std::vector<uint8_t> &)>;

/**
 * Loads compatible prepared tensors before their portable source and persists misses.
 */
class ONNX_LIGHT_CORE_API PreparedTensorCache {
public:
  PreparedTensorCache() = default;
  ~PreparedTensorCache();
  PreparedTensorCache(const PreparedTensorCache &) = delete;
  PreparedTensorCache &operator=(const PreparedTensorCache &) = delete;

  /**
   * Publishes a compatible cached payload or prepares the portable source on a miss.
   *
   * Cache persistence starts only after the complete prepared value is published.
   */
  PreparedTensorLoadResult LoadOrPrepare(const std::filesystem::path &cache_path,
                                         const PreparedTensorMetadata &required_metadata,
                                         const PreparedSourceLoader &load_source,
                                         const PreparedTensorPrepacker &prepack,
                                         const PreparedTensorPublisher &publish);

  /** Waits for all pending atomic cache writes. */
  void WaitForBackgroundWrites();

  /** Waits for pending writes and returns their diagnostics. */
  std::vector<std::string> TakePersistenceDiagnostics();

private:
  struct PersistenceRequest {
    std::filesystem::path cache_path;
    PreparedTensorMetadata metadata;
    std::vector<uint8_t> payload;
  };

  void PersistInBackground(std::filesystem::path cache_path, PreparedTensorMetadata metadata,
                           std::vector<uint8_t> payload);
  void WriterLoop();

  std::mutex mutex_;
  std::condition_variable writer_ready_;
  std::deque<PersistenceRequest> pending_;
  std::thread writer_;
  size_t pending_writes_ = 0;
  bool stop_writer_ = false;
  std::vector<std::string> persistence_diagnostics_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
