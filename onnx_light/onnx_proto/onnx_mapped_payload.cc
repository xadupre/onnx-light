// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#include "onnx_mapped_payload.h"
#include "stream.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

#if !defined(_WIN32)
#include <unistd.h>
#else
#define NOMINMAX
#include <windows.h>
#endif

namespace ONNX_LIGHT_NAMESPACE {

namespace {

// Returns the OS memory-mapping granularity (POSIX page size / Windows allocation
// granularity). mmap_file_as_shared_ptr (stream.cc) maps whole files at offset 0, so a
// mapped payload's base address always has this alignment.
size_t SystemPageAlignment() {
#if !defined(_WIN32)
  static const size_t page_size = [] {
    const long value = ::sysconf(_SC_PAGESIZE);
    return value > 0 ? static_cast<size_t>(value) : size_t{4096};
  }();
#else
  static const size_t page_size = [] {
    SYSTEM_INFO info;
    ::GetSystemInfo(&info);
    return info.dwPageSize > 0 ? static_cast<size_t>(info.dwPageSize) : size_t{4096};
  }();
#endif
  return page_size;
}

// Returns the alignment of (page-aligned base address + offset): the largest power of
// two that divides both the page alignment and offset.
size_t AlignmentForOffset(size_t base_alignment, int64_t offset) {
  if (offset == 0 || base_alignment == 0) {
    return base_alignment;
  }
  const uint64_t off = static_cast<uint64_t>(offset);
  const uint64_t low_bit = off & (~off + 1ULL);
  return static_cast<size_t>(std::min<uint64_t>(base_alignment, low_bit));
}

} // namespace

std::string PayloadIdentity::string() const {
  std::ostringstream oss;
  oss << "PayloadIdentity(source_path=" << source_path << ", offset=" << offset << ", size=" << size
      << ", generation=" << generation << ")";
  return oss.str();
}

FinalDestinationReadDescriptor::FinalDestinationReadDescriptor(std::string source_path,
                                                               int64_t offset, int64_t size,
                                                               PayloadIdentity identity)
    : source_path_(std::move(source_path)), offset_(offset), size_(size),
      identity_(std::move(identity)) {}

void FinalDestinationReadDescriptor::ReadInto(void *destination) const {
  EXT_ENFORCE(size_ == 0 || destination != nullptr,
              "FinalDestinationReadDescriptor::ReadInto: destination is null for size=", size_);
  if (size_ == 0) {
    return;
  }
  std::error_code ec;
  const auto current_size = std::filesystem::file_size(source_path_, ec);
  EXT_ENFORCE(!ec, "FinalDestinationReadDescriptor::ReadInto: cannot stat source file '",
              source_path_, "': ", ec.message());
  EXT_ENFORCE(static_cast<int64_t>(current_size) >= offset_ + size_,
              "FinalDestinationReadDescriptor::ReadInto: source file '", source_path_,
              "' was truncated: required end=", offset_ + size_,
              ", current size=", static_cast<int64_t>(current_size));
  std::ifstream file(source_path_, std::ios::binary);
  EXT_ENFORCE(file.is_open(), "FinalDestinationReadDescriptor::ReadInto: cannot open source file '",
              source_path_, "'");
  file.seekg(static_cast<std::streamoff>(offset_));
  EXT_ENFORCE(!file.fail(),
              "FinalDestinationReadDescriptor::ReadInto: cannot seek in source file '",
              source_path_, "' to offset=", offset_);
  file.read(reinterpret_cast<char *>(destination), static_cast<std::streamsize>(size_));
  EXT_ENFORCE(file.gcount() == static_cast<std::streamsize>(size_),
              "FinalDestinationReadDescriptor::ReadInto: short read from source file '",
              source_path_, "': expected=", size_, ", got=", static_cast<int64_t>(file.gcount()));
}

MappedPayloadSource::MappedPayloadSource(std::string base_dir) : base_dir_(std::move(base_dir)) {
  EXT_ENFORCE(!base_dir_.empty(), "MappedPayloadSource: base_dir cannot be empty.");
}

std::string
MappedPayloadSource::ResolveAndValidate(const std::string &relative_or_absolute_path) const {
  EXT_ENFORCE(!relative_or_absolute_path.empty(), "MappedPayloadSource: path cannot be empty.");
  std::filesystem::path candidate(relative_or_absolute_path);
  const std::filesystem::path base(base_dir_);
  if (!candidate.is_absolute()) {
    candidate = base / candidate;
  }
  utils::validate_external_weights_read_path(candidate, base);
  std::error_code ec;
  const std::filesystem::path canonical = std::filesystem::weakly_canonical(candidate, ec);
  EXT_ENFORCE(!ec, "MappedPayloadSource: could not canonicalize path '", candidate.string(),
              "': ", ec.message());
  return canonical.string();
}

MappedPayloadSource::CachedMapping
MappedPayloadSource::EnsureMapping(const std::string &canonical_path) {
  std::error_code ec;
  const auto file_size = std::filesystem::file_size(canonical_path, ec);
  EXT_ENFORCE(!ec, "MappedPayloadSource: cannot stat source file '", canonical_path,
              "': ", ec.message());
  const int64_t current_size = static_cast<int64_t>(file_size);

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = mappings_.find(canonical_path);
  if (it != mappings_.end() && it->second.size == current_size) {
    // Cache hit: the file did not change size since it was last mapped. A same-size
    // in-place replacement is out of scope for a stat-based check; callers that need
    // that guarantee should call ReleaseCachedMappings() before Borrow().
    return it->second;
  }

  CachedMapping mapping;
  mapping.size = current_size;
  mapping.generation = next_generation_++;
  mapping.data = utils::mmap_file_as_shared_ptr(canonical_path, current_size);
  mappings_[canonical_path] = mapping;
  return mapping;
}

MappedPayload MappedPayloadSource::Borrow(const std::string &relative_or_absolute_path,
                                          int64_t offset, int64_t size) {
  EXT_ENFORCE(offset >= 0, "MappedPayloadSource::Borrow: offset must be >= 0, got ", offset);
  EXT_ENFORCE(size >= 0, "MappedPayloadSource::Borrow: size must be >= 0, got ", size);
  const std::string canonical_path = ResolveAndValidate(relative_or_absolute_path);
  const CachedMapping mapping = EnsureMapping(canonical_path);
  EXT_ENFORCE(offset <= mapping.size && size <= mapping.size - offset,
              "MappedPayloadSource::Borrow: range is out of bounds for file '", canonical_path,
              "': offset=", offset, ", size=", size, ", file_size=", mapping.size);

  MappedPayload payload;
  payload.size = static_cast<size_t>(size);
  payload.identity.source_path = canonical_path;
  payload.identity.offset = offset;
  payload.identity.size = size;
  payload.identity.generation = mapping.generation;
  if (size == 0) {
    payload.data = nullptr;
    payload.alignment = 0;
    return payload;
  }
  payload.data = mapping.data.get() + offset;
  payload.alignment = AlignmentForOffset(SystemPageAlignment(), offset);
  payload.owner = std::static_pointer_cast<void>(mapping.data);
  return payload;
}

FinalDestinationReadDescriptor
MappedPayloadSource::DescribeFinalDestinationRead(const std::string &relative_or_absolute_path,
                                                  int64_t offset, int64_t size) const {
  EXT_ENFORCE(offset >= 0,
              "MappedPayloadSource::DescribeFinalDestinationRead: offset must be >= 0, got ",
              offset);
  EXT_ENFORCE(size >= 0,
              "MappedPayloadSource::DescribeFinalDestinationRead: size must be >= 0, got ", size);
  const std::string canonical_path = ResolveAndValidate(relative_or_absolute_path);
  std::error_code ec;
  const auto file_size = std::filesystem::file_size(canonical_path, ec);
  EXT_ENFORCE(!ec, "MappedPayloadSource::DescribeFinalDestinationRead: cannot stat source file '",
              canonical_path, "': ", ec.message());
  const int64_t current_size = static_cast<int64_t>(file_size);
  EXT_ENFORCE(
      offset <= current_size && size <= current_size - offset,
      "MappedPayloadSource::DescribeFinalDestinationRead: range is out of bounds for file '",
      canonical_path, "': offset=", offset, ", size=", size, ", file_size=", current_size);

  PayloadIdentity identity;
  identity.source_path = canonical_path;
  identity.offset = offset;
  identity.size = size;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = mappings_.find(canonical_path);
    identity.generation = it != mappings_.end() ? it->second.generation : 0;
  }
  return FinalDestinationReadDescriptor(canonical_path, offset, size, identity);
}

void MappedPayloadSource::ReleaseCachedMappings() {
  std::lock_guard<std::mutex> lock(mutex_);
  mappings_.clear();
}

} // namespace ONNX_LIGHT_NAMESPACE
