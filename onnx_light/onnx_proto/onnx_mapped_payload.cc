// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#include "onnx_mapped_payload.h"
#include "external_data_internal.h"
#include "stream.h"

#include <algorithm>
#include <filesystem>
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
// mapped payload's base address always has this alignment. On Windows the mapping's
// base-address alignment is governed by dwAllocationGranularity (typically 64 KiB),
// not dwPageSize (typically 4 KiB): using the page size would overstate the guarantee.
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
    return info.dwAllocationGranularity > 0 ? static_cast<size_t>(info.dwAllocationGranularity)
                                            : size_t{65536};
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
    : FinalDestinationReadDescriptor(source_path, offset, size, std::move(identity),
                                     utils::detail::OpenExternalDataFile(
                                         std::filesystem::path(source_path).parent_path().string(),
                                         std::filesystem::path(source_path).filename().string())) {}

FinalDestinationReadDescriptor::FinalDestinationReadDescriptor(std::string source_path,
                                                               int64_t offset, int64_t size,
                                                               PayloadIdentity identity,
                                                               std::shared_ptr<int> file)
    : source_path_(std::move(source_path)), offset_(offset), size_(size),
      identity_(std::move(identity)), file_(std::move(file)) {
  EXT_ENFORCE(offset_ >= 0 && size_ >= 0, "FinalDestinationReadDescriptor: invalid range.");
  EXT_ENFORCE(file_ != nullptr, "FinalDestinationReadDescriptor: source is not open.");
  const int64_t current_size = utils::detail::ExternalDataFileSize(*file_);
  EXT_ENFORCE(offset_ <= current_size && size_ <= current_size - offset_,
              "FinalDestinationReadDescriptor: range is out of bounds.");
}

void FinalDestinationReadDescriptor::ReadInto(void *destination) const {
  EXT_ENFORCE(size_ == 0 || destination != nullptr,
              "FinalDestinationReadDescriptor::ReadInto: destination is null for size=", size_);
  if (size_ == 0) {
    return;
  }
  EXT_ENFORCE(file_ != nullptr, "FinalDestinationReadDescriptor::ReadInto: source is not open.");
  const int64_t current_size = utils::detail::ExternalDataFileSize(*file_);
  EXT_ENFORCE(offset_ <= current_size && size_ <= current_size - offset_,
              "FinalDestinationReadDescriptor::ReadInto: source file '", source_path_,
              "' was truncated: required end=", offset_ + size_, ", current size=", current_size);
  utils::detail::ReadExternalDataFile(*file_, destination, size_, offset_);
}

MappedPayloadSource::MappedPayloadSource(std::string base_dir) : base_dir_(std::move(base_dir)) {
  EXT_ENFORCE(!base_dir_.empty(), "MappedPayloadSource: base_dir cannot be empty.");
}

std::string
MappedPayloadSource::ResolveAndValidate(const std::string &relative_or_absolute_path) const {
  EXT_ENFORCE(!relative_or_absolute_path.empty(), "MappedPayloadSource: path cannot be empty.");
  std::filesystem::path candidate(relative_or_absolute_path);
  const std::filesystem::path base = std::filesystem::absolute(base_dir_).lexically_normal();
  if (!candidate.is_absolute()) {
    candidate = base / candidate;
  }
  utils::validate_external_weights_read_path(candidate, base);
  std::error_code ec;
  const std::filesystem::path canonical_base = std::filesystem::weakly_canonical(base, ec);
  EXT_ENFORCE(!ec, "MappedPayloadSource: could not canonicalize base '", base.string(),
              "': ", ec.message());
  return (canonical_base / utils::detail::relative_external_data_path(candidate, base)).string();
}

std::shared_ptr<int> MappedPayloadSource::OpenFile(const std::string &source_path) const {
  const auto base = std::filesystem::weakly_canonical(std::filesystem::absolute(base_dir_));
  return utils::detail::OpenExternalDataFile(
      base.string(), std::filesystem::path(source_path).lexically_relative(base).string());
}

MappedPayloadSource::CachedMapping
MappedPayloadSource::EnsureMapping(const std::string &canonical_path) {
  // The stat + cache-lookup + mmap sequence runs entirely under the lock so two
  // threads racing to map the same not-yet-cached file cannot both create a
  // mapping (which would hand out two different owners/generations for the
  // same file content).
  std::lock_guard<std::mutex> lock(mutex_);
  const auto file = OpenFile(canonical_path);
  const int64_t current_size = utils::detail::ExternalDataFileSize(*file);

  auto it = mappings_.find(canonical_path);
  if (it != mappings_.end() && it->second.size == current_size) {
    // Cache hit: the file did not change size since it was last mapped. A same-size
    // in-place replacement is out of scope for a stat-based check; callers that need
    // that guarantee should call ReleaseCachedMappings() before Borrow().
    return it->second;
  }

  CachedMapping mapping;
  mapping.file = file;
  mapping.size = current_size;
  mapping.generation = next_generation_++;
  mapping.data = utils::detail::MapExternalDataFile(*file);
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
  const auto file = OpenFile(canonical_path);
  const int64_t current_size = utils::detail::ExternalDataFileSize(*file);
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
  return FinalDestinationReadDescriptor(canonical_path, offset, size, identity, file);
}

void MappedPayloadSource::ReleaseCachedMappings() {
  std::lock_guard<std::mutex> lock(mutex_);
  mappings_.clear();
}

} // namespace ONNX_LIGHT_NAMESPACE
