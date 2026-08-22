// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "onnx_light_helpers.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE {

/** Stable identity of one payload view backed by a source-file byte range.
 *
 *  Two identities compare equal only when they name the same canonical source
 *  path, the same byte range, and the same file *generation*. The generation
 *  is bumped by MappedPayloadSource every time it (re)maps a path, so a file
 *  replaced or truncated between two requests never silently reuses a stale
 *  identity: the new mapping gets a new generation even though the path is
 *  unchanged. */
struct ONNX_LIGHT_PROTO_API PayloadIdentity {
  std::string source_path;
  int64_t offset = 0;
  int64_t size = 0;
  uint64_t generation = 0;

  bool operator==(const PayloadIdentity &other) const noexcept {
    return offset == other.offset && size == other.size && generation == other.generation &&
           source_path == other.source_path;
  }
  bool operator!=(const PayloadIdentity &other) const noexcept { return !(*this == other); }

  /** Returns a human-readable representation for diagnostics. */
  std::string string() const;
};

/** ORT-independent, read-only view of memory owned by onnx-light.
 *
 *  Lifetime: `owner` keeps the backing storage (a memory mapping) alive.
 *  `data` stays valid for as long as at least one copy of `owner` is alive;
 *  a caller that keeps `data` beyond the MappedPayload value itself must keep
 *  a copy of `owner` too. Destroying every copy of `owner` releases the
 *  mapping (unmaps the region); this is the only way a mapping is released.
 *  Immutability: `data` refers to read-only memory (POSIX mmap uses
 *  PROT_READ, Windows uses FILE_MAP_READ). Writing through `data` is
 *  undefined behavior.
 *  Alignment: `alignment` is the guaranteed byte alignment of `data`. A
 *  memory-mapped payload starts at a page-aligned base address combined with
 *  the requested byte offset, so `alignment` reports the largest power of two
 *  that evenly divides the mapped page alignment and the offset; it is 0 when
 *  `data` is null (an empty payload).
 *  Identity: see PayloadIdentity. */
struct ONNX_LIGHT_PROTO_API MappedPayload {
  const void *data = nullptr;
  size_t size = 0;
  size_t alignment = 0;
  std::shared_ptr<void> owner;
  PayloadIdentity identity;
};

/** Validated descriptor for a source range that must be read directly into a
 *  caller-owned final allocation instead of being borrowed as a
 *  MappedPayload -- for example because the destination requires writable
 *  memory, a different alignment, relocation, or another physical layout
 *  than the mapped source.
 *
 *  Error semantics: constructing a descriptor (via
 *  MappedPayloadSource::DescribeFinalDestinationRead) validates that the
 *  source path is confined to the source's base directory, is not a
 *  symlink, has at most one hard link, and that the requested range fits
 *  inside the file's size at validation time. ReadInto() re-validates the
 *  file has not shrunk since and throws rather than silently truncating the
 *  read. onnx-light never allocates an intermediate whole-tensor buffer to
 *  service this path: ReadInto() reads directly from the source file into
 *  `destination`. */
class ONNX_LIGHT_PROTO_API FinalDestinationReadDescriptor {
public:
  FinalDestinationReadDescriptor() = default;
  FinalDestinationReadDescriptor(std::string source_path, int64_t offset, int64_t size,
                                 PayloadIdentity identity);

  const std::string &source_path() const { return source_path_; }
  int64_t offset() const { return offset_; }
  int64_t size() const { return size_; }
  const PayloadIdentity &identity() const { return identity_; }

  /** Reads size() bytes at offset() from source_path() directly into
   *  *destination*, which must have room for at least size() bytes. Throws
   *  if the file cannot be opened, is shorter than required (e.g. a
   *  truncation raced with validation), or a short/failed read occurs. */
  void ReadInto(void *destination) const;

private:
  std::string source_path_;
  int64_t offset_ = 0;
  int64_t size_ = 0;
  PayloadIdentity identity_;
};

/** Confines external-payload requests to one base directory, shares one
 *  memory mapping per canonical source file across concurrent borrows, and
 *  exposes both a MappedPayload (borrow) and a FinalDestinationReadDescriptor
 *  (validated direct read) view over the same byte range. The caller decides
 *  which view to use; onnx-light does not decide eligibility for borrowing.
 *
 *  Thread-safety: Borrow() and DescribeFinalDestinationRead() may be called
 *  concurrently, including for the same source file; the shared mapping
 *  cache is protected by an internal mutex, and concurrently returned
 *  MappedPayload views over the same file share one owner.
 *  Path confinement: every source path is resolved relative to *base_dir*
 *  (unless already absolute) and validated with the same rules applied to
 *  external tensor data reads: no symlinks, no additional hard links, and
 *  the canonicalized path must resolve inside *base_dir*. */
class ONNX_LIGHT_PROTO_API MappedPayloadSource {
public:
  explicit MappedPayloadSource(std::string base_dir);

  /** Returns the base directory every requested path is confined to. */
  const std::string &base_dir() const { return base_dir_; }

  /** Returns a shared, read-only memory-mapped view of
   *  `[offset, offset + size)` inside *relative_or_absolute_path*. Throws on
   *  confinement violations, a negative offset/size, or a range that does
   *  not fit inside the file. The returned `alignment` reflects the actual
   *  alignment of `data`; compare it against a required alignment to decide
   *  whether the range is eligible to be borrowed as-is. */
  MappedPayload Borrow(const std::string &relative_or_absolute_path, int64_t offset, int64_t size);

  /** Returns a validated final-destination descriptor for the same range
   *  without requiring the path to be memory-mapped. Use this whenever the
   *  caller cannot borrow the mapped view (Borrow()'s alignment is
   *  insufficient, the destination needs writable memory, or another
   *  physical layout is required). */
  FinalDestinationReadDescriptor
  DescribeFinalDestinationRead(const std::string &relative_or_absolute_path, int64_t offset,
                               int64_t size) const;

  /** Drops every cached memory mapping. Previously returned MappedPayload
   *  values keep their own `owner` reference and remain valid; a later
   *  Borrow() call for the same path remaps the file and observes a new
   *  generation and identity. */
  void ReleaseCachedMappings();

private:
  struct CachedMapping {
    std::shared_ptr<uint8_t> data;
    int64_t size = 0;
    uint64_t generation = 0;
  };

  std::string ResolveAndValidate(const std::string &relative_or_absolute_path) const;
  CachedMapping EnsureMapping(const std::string &canonical_path);

  std::string base_dir_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, CachedMapping> mappings_;
  uint64_t next_generation_ = 1;
};

} // namespace ONNX_LIGHT_NAMESPACE
