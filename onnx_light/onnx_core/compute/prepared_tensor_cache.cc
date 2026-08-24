// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "prepared_tensor_cache.h"

#include "onnx_proto/blake3/blake3_hash.h"

#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <limits>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

constexpr std::array<char, 8> kMagic{'O', 'N', 'L', 'P', 'T', 'C', '1', '\0'};
constexpr uint32_t kSchemaVersion = 1;
constexpr uint32_t kMaximumMetadataLength = 1U << 20;
std::atomic<uint64_t> next_temporary_id{0};

uint64_t ProcessId() {
#if defined(_WIN32)
  return static_cast<uint64_t>(::GetCurrentProcessId());
#else
  return static_cast<uint64_t>(::getpid());
#endif
}

uint64_t PayloadDigest(const std::vector<uint8_t> &payload) {
  utils::Blake3Hasher hasher;
  hasher.Update(payload.data(), payload.size());
  return static_cast<uint64_t>(hasher.Finalize64());
}

template <typename T> bool ReadScalar(std::istream &input, T &value) {
  input.read(reinterpret_cast<char *>(&value), sizeof(value));
  return input.good();
}

template <typename T> void WriteScalar(std::ostream &output, T value) {
  output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

bool ReadString(std::istream &input, std::string &value) {
  uint32_t size = 0;
  if (!ReadScalar(input, size) || size > kMaximumMetadataLength) {
    return false;
  }
  value.resize(size);
  input.read(value.data(), static_cast<std::streamsize>(size));
  return input.good();
}

void WriteString(std::ostream &output, const std::string &value) {
  EXT_ENFORCE(value.size() <= kMaximumMetadataLength,
              "Prepared-cache metadata exceeds the supported size.");
  WriteScalar(output, static_cast<uint32_t>(value.size()));
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
}

struct CacheEntry {
  PreparedTensorMetadata metadata;
  std::vector<uint8_t> payload;
};

bool ReadCacheEntry(const std::filesystem::path &path, CacheEntry &entry, std::string &diagnostic) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    diagnostic = "prepared cache entry is absent";
    return false;
  }
  std::array<char, kMagic.size()> magic{};
  uint32_t schema_version = 0;
  uint64_t required_isa = 0;
  uint64_t payload_size = 0;
  uint64_t payload_digest = 0;
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input.good() || magic != kMagic || !ReadScalar(input, schema_version) ||
      schema_version != kSchemaVersion || !ReadString(input, entry.metadata.source_digest) ||
      !ReadString(input, entry.metadata.architecture) || !ReadScalar(input, required_isa) ||
      !ReadString(input, entry.metadata.runtime) ||
      !ReadString(input, entry.metadata.runtime_version) ||
      !ReadString(input, entry.metadata.kernel_layout) ||
      !ReadString(input, entry.metadata.format_version) || !ReadScalar(input, payload_size) ||
      !ReadScalar(input, payload_digest) ||
      payload_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      payload_size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    diagnostic = "prepared cache metadata is corrupt";
    return false;
  }
  const std::streampos payload_position = input.tellg();
  input.seekg(0, std::ios::end);
  const std::streampos file_end = input.tellg();
  if (payload_position < 0 || file_end < payload_position ||
      static_cast<uint64_t>(file_end - payload_position) != payload_size) {
    diagnostic = "prepared cache payload length is corrupt";
    return false;
  }
  input.seekg(payload_position);
  entry.metadata.required_isa = platform::CpuFeatureSet(required_isa);
  entry.payload.resize(static_cast<size_t>(payload_size));
  input.read(reinterpret_cast<char *>(entry.payload.data()),
             static_cast<std::streamsize>(entry.payload.size()));
  if (static_cast<size_t>(input.gcount()) != entry.payload.size() ||
      PayloadDigest(entry.payload) != payload_digest) {
    diagnostic = "prepared cache payload is corrupt";
    return false;
  }
  return true;
}

PreparedCacheMissReason CheckCompatibility(const PreparedTensorMetadata &cached,
                                           const PreparedTensorMetadata &required,
                                           std::string &diagnostic) {
  if (cached.source_digest != required.source_digest) {
    diagnostic = "prepared cache source digest is stale";
    return PreparedCacheMissReason::kStaleSource;
  }
  if (cached.architecture != required.architecture) {
    diagnostic = "prepared cache architecture is incompatible";
    return PreparedCacheMissReason::kWrongArchitecture;
  }
  if (!required.required_isa.ContainsAll(cached.required_isa)) {
    diagnostic = "prepared cache requires an unavailable ISA";
    return PreparedCacheMissReason::kWrongIsa;
  }
  if (cached.runtime != required.runtime || cached.runtime_version != required.runtime_version) {
    diagnostic = "prepared cache runtime is incompatible";
    return PreparedCacheMissReason::kWrongRuntime;
  }
  if (cached.kernel_layout != required.kernel_layout) {
    diagnostic = "prepared cache kernel layout is incompatible";
    return PreparedCacheMissReason::kWrongLayout;
  }
  if (cached.format_version != required.format_version) {
    diagnostic = "prepared cache format is incompatible";
    return PreparedCacheMissReason::kWrongFormat;
  }
  return PreparedCacheMissReason::kNone;
}

bool WriteCacheEntry(const std::filesystem::path &path, const PreparedTensorMetadata &metadata,
                     const std::vector<uint8_t> &payload, std::string &diagnostic) {
  std::error_code error;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
      diagnostic = "Unable to create prepared-cache directory: " + error.message();
      return false;
    }
  }
  std::filesystem::path temporary = path;
  temporary += ".tmp." + std::to_string(ProcessId()) + "." + std::to_string(++next_temporary_id);
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      diagnostic = "Unable to write temporary prepared-cache entry.";
      return false;
    }
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    WriteScalar(output, kSchemaVersion);
    WriteString(output, metadata.source_digest);
    WriteString(output, metadata.architecture);
    WriteScalar(output, metadata.required_isa.bits());
    WriteString(output, metadata.runtime);
    WriteString(output, metadata.runtime_version);
    WriteString(output, metadata.kernel_layout);
    WriteString(output, metadata.format_version);
    WriteScalar(output, static_cast<uint64_t>(payload.size()));
    WriteScalar(output, PayloadDigest(payload));
    output.write(reinterpret_cast<const char *>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    output.flush();
    if (!output) {
      diagnostic = "Unable to flush temporary prepared-cache entry.";
      std::filesystem::remove(temporary, error);
      return false;
    }
  }

#if defined(_WIN32)
  HANDLE handle =
      ::CreateFileW(temporary.native().c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE || !::FlushFileBuffers(handle)) {
    if (handle != INVALID_HANDLE_VALUE) {
      ::CloseHandle(handle);
    }
    diagnostic = "Unable to synchronize temporary prepared-cache entry.";
    std::filesystem::remove(temporary, error);
    return false;
  }
  ::CloseHandle(handle);
  if (!::MoveFileExW(temporary.native().c_str(), path.native().c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    diagnostic = "Unable to atomically replace prepared-cache entry.";
    std::filesystem::remove(temporary, error);
    return false;
  }
#else
  const int descriptor = ::open(temporary.c_str(), O_RDONLY);
  if (descriptor < 0 || ::fsync(descriptor) != 0) {
    if (descriptor >= 0) {
      ::close(descriptor);
    }
    diagnostic = "Unable to synchronize temporary prepared-cache entry.";
    std::filesystem::remove(temporary, error);
    return false;
  }
  ::close(descriptor);
  std::filesystem::rename(temporary, path, error);
  if (error) {
    diagnostic = "Unable to atomically replace prepared-cache entry: " + error.message();
    std::filesystem::remove(temporary, error);
    return false;
  }
  const std::filesystem::path directory = path.parent_path().empty() ? "." : path.parent_path();
  const int directory_descriptor = ::open(directory.c_str(), O_RDONLY);
  if (directory_descriptor >= 0) {
    (void)::fsync(directory_descriptor);
    ::close(directory_descriptor);
  }
#endif
  return true;
}

} // namespace

PreparedTensorCache::~PreparedTensorCache() {
  WaitForBackgroundWrites();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_writer_ = true;
  }
  writer_ready_.notify_one();
  if (writer_.joinable()) {
    writer_.join();
  }
}

PreparedTensorLoadResult PreparedTensorCache::LoadOrPrepare(
    const std::filesystem::path &cache_path, const PreparedTensorMetadata &required_metadata,
    const PreparedSourceLoader &load_source, const PreparedTensorPrepacker &prepack,
    const PreparedTensorPublisher &publish) {
  EXT_ENFORCE(!required_metadata.source_digest.empty(),
              "A prepared tensor requires a source digest.");
  EXT_ENFORCE(!required_metadata.architecture.empty(),
              "A prepared tensor requires a CPU architecture.");
  EXT_ENFORCE(!required_metadata.runtime.empty(), "A prepared tensor requires a runtime.");
  EXT_ENFORCE(!required_metadata.runtime_version.empty(),
              "A prepared tensor requires a runtime version.");
  EXT_ENFORCE(!required_metadata.kernel_layout.empty(),
              "A prepared tensor requires a kernel layout.");
  EXT_ENFORCE(!required_metadata.format_version.empty(),
              "A prepared tensor requires a format version.");
  const auto validate_metadata_size = [](const std::string &value) {
    EXT_ENFORCE(value.size() <= kMaximumMetadataLength,
                "Prepared-cache metadata exceeds the supported size.");
  };
  validate_metadata_size(required_metadata.source_digest);
  validate_metadata_size(required_metadata.architecture);
  validate_metadata_size(required_metadata.runtime);
  validate_metadata_size(required_metadata.runtime_version);
  validate_metadata_size(required_metadata.kernel_layout);
  validate_metadata_size(required_metadata.format_version);

  PreparedTensorLoadResult result;
  CacheEntry cached;
  if (ReadCacheEntry(cache_path, cached, result.diagnostic)) {
    result.miss_reason = CheckCompatibility(cached.metadata, required_metadata, result.diagnostic);
    if (result.miss_reason == PreparedCacheMissReason::kNone) {
      publish(cached.payload);
      result.cache_hit = true;
      return result;
    }
  } else {
    std::error_code error;
    result.miss_reason = std::filesystem::exists(cache_path, error)
                             ? PreparedCacheMissReason::kCorrupt
                             : PreparedCacheMissReason::kNotFound;
  }

  std::vector<uint8_t> prepared = prepack(load_source());
  publish(prepared);
  PersistInBackground(cache_path, required_metadata, std::move(prepared));
  return result;
}

void PreparedTensorCache::PersistInBackground(std::filesystem::path cache_path,
                                              PreparedTensorMetadata metadata,
                                              std::vector<uint8_t> payload) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(
        PersistenceRequest{std::move(cache_path), std::move(metadata), std::move(payload)});
    ++pending_writes_;
    if (!writer_.joinable()) {
      writer_ = std::thread([this]() { WriterLoop(); });
    }
  }
  writer_ready_.notify_one();
}

void PreparedTensorCache::WriterLoop() {
  while (true) {
    PersistenceRequest request;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      writer_ready_.wait(lock, [this]() { return stop_writer_ || !pending_.empty(); });
      if (stop_writer_ && pending_.empty()) {
        return;
      }
      request = std::move(pending_.front());
      pending_.pop_front();
    }
    std::string diagnostic;
    if (!WriteCacheEntry(request.cache_path, request.metadata, request.payload, diagnostic)) {
      std::lock_guard<std::mutex> lock(mutex_);
      persistence_diagnostics_.push_back(std::move(diagnostic));
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      --pending_writes_;
    }
    writer_ready_.notify_all();
  }
}

void PreparedTensorCache::WaitForBackgroundWrites() {
  std::unique_lock<std::mutex> lock(mutex_);
  writer_ready_.wait(lock, [this]() { return pending_writes_ == 0; });
}

std::vector<std::string> PreparedTensorCache::TakePersistenceDiagnostics() {
  WaitForBackgroundWrites();
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> diagnostics;
  diagnostics.swap(persistence_diagnostics_);
  return diagnostics;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
