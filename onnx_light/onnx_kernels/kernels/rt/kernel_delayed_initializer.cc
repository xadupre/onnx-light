// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/rt/include_rt_kernels.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

/// Validates DelayedInitializer attributes that are independent from file contents.
void ValidateAttributes(const DelayedInitializer::Attributes &attrs) {
  EXT_ENFORCE_INVALID(attrs.load_device == "cpu" || attrs.load_device == "file",
                      "kernel::DelayedInitializer load_device must be 'cpu' or 'file', got '",
                      attrs.load_device, "'.");
  EXT_ENFORCE_INVALID(attrs.runtime_device == "cpu",
                      "kernel::DelayedInitializer runtime_device must be 'cpu', got '",
                      attrs.runtime_device, "'.");
  EXT_ENFORCE_INVALID(!attrs.filename.empty(),
                      "kernel::DelayedInitializer filename must not be empty.");
  EXT_ENFORCE_INVALID(attrs.offset >= 0,
                      "kernel::DelayedInitializer offset must be non-negative, got ", attrs.offset,
                      ".");
  EXT_ENFORCE_INVALID(attrs.dtype != static_cast<int32_t>(DataType::STRING),
                      "kernel::DelayedInitializer does not support STRING tensors.");
}

} // namespace

DelayedInitializer::DelayedInitializer(const KernelContext &ctx, Attributes attrs)
    : KernelBase(ctx), attrs_(std::move(attrs)) {
  ValidateAttributes(attrs_);
  if (attrs_.load_device == "cpu") {
    loaded_bytes_ = LoadBytes(attrs_);
  }
}

Tensor DelayedInitializer::operator()() const {
  if (attrs_.load_device == "cpu") {
    return Tensor("", attrs_.dtype, attrs_.shape, loaded_bytes_);
  }
  return Tensor("", attrs_.dtype, attrs_.shape, LoadBytes(attrs_));
}

/// Computes the total number of elements described by a shape.
int64_t DelayedInitializer::ComputeElementCount(const std::vector<int64_t> &shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    EXT_ENFORCE_INVALID(dim >= 0,
                        "kernel::DelayedInitializer shape dimensions must be non-negative.");
    if (dim != 0) {
      EXT_ENFORCE_INVALID(count <= std::numeric_limits<int64_t>::max() / dim,
                          "kernel::DelayedInitializer shape is too large.");
    }
    count *= dim;
  }
  return count;
}

/// Loads the raw tensor payload described by the DelayedInitializer attributes.
std::vector<uint8_t> DelayedInitializer::LoadBytes(const Attributes &attrs) {
  const int64_t element_count = ComputeElementCount(attrs.shape);
  const size_t byte_count = PackedByteSize(attrs.dtype, element_count);
  std::vector<uint8_t> bytes(byte_count);
  if (byte_count == 0) {
    return bytes;
  }

  const std::filesystem::path path(attrs.filename);
  EXT_ENFORCE_INVALID(std::filesystem::is_regular_file(path),
                      "kernel::DelayedInitializer filename is not a regular file: '",
                      attrs.filename, "'.");

  errno = 0;
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  const int open_errno = errno;
  EXT_ENFORCE_INVALID(stream.good(), "kernel::DelayedInitializer unable to open file '",
                      attrs.filename, "': ", std::strerror(open_errno), ".");

  const std::streamoff file_size = stream.tellg();
  EXT_ENFORCE_INVALID(file_size >= 0,
                      "kernel::DelayedInitializer unable to determine file size for '",
                      attrs.filename, "'.");
  const uint64_t offset = static_cast<uint64_t>(attrs.offset);
  EXT_ENFORCE_INVALID(offset <= static_cast<uint64_t>(file_size),
                      "kernel::DelayedInitializer offset ", attrs.offset,
                      " exceeds file size for '", attrs.filename, "'.");
  const uint64_t remaining = static_cast<uint64_t>(file_size) - offset;
  EXT_ENFORCE_INVALID(byte_count <= remaining, "kernel::DelayedInitializer file '", attrs.filename,
                      "' does not contain ", byte_count, " bytes at offset ", attrs.offset, ".");

  stream.seekg(static_cast<std::streamoff>(attrs.offset), std::ios::beg);
  EXT_ENFORCE_INVALID(stream.good(), "kernel::DelayedInitializer seek failed for '", attrs.filename,
                      "'.");
  stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(byte_count));
  EXT_ENFORCE_INVALID(stream.gcount() == static_cast<std::streamsize>(byte_count),
                      "kernel::DelayedInitializer read failed for '", attrs.filename,
                      "': expected ", byte_count, " bytes, got ", stream.gcount(), ".");
  return bytes;
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
