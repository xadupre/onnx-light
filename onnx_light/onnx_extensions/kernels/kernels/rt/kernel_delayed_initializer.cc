// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/rt/include_rt_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

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
    loaded_bytes_ = LoadBytes(attrs_, ctx_.allocator);
  }
}

Tensor DelayedInitializer::operator()(RuntimeContext *rt) const {
  // Prefer the runtime allocator when one is available, otherwise fall back to
  // the construction-time allocator. When neither is set, MakeOutputTensor and
  // LoadBytes transparently fall back to inline std::vector storage.
  RawBufferAllocator *allocator =
      (rt != nullptr && rt->allocator() != nullptr) ? rt->allocator() : ctx_.allocator;
  if (attrs_.load_device == "cpu") {
    const size_t byte_count = loaded_bytes_.size_bytes();
    Tensor output = MakeOutputTensor(attrs_.dtype, attrs_.shape, byte_count, allocator);
    if (byte_count != 0) {
      std::memcpy(output.mutable_bytes(), loaded_bytes_.bytes(), byte_count);
    }
    return output;
  }
  return LoadBytes(attrs_, allocator);
}

/// Computes the total number of elements described by a shape.
int64_t DelayedInitializer::ComputeElementCount(const onnx_kernels::Shape &shape) {
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
Tensor DelayedInitializer::LoadBytes(const Attributes &attrs, RawBufferAllocator *allocator) {
  const int64_t element_count = ComputeElementCount(attrs.shape);
  const size_t byte_count = PackedByteSize(attrs.dtype, element_count);
  Tensor t = MakeOutputTensor(attrs.dtype, attrs.shape, byte_count, allocator);
  LoadBytesInto(attrs, t.mutable_bytes(), byte_count);
  return t;
}

/// Loads bytes from ``attrs.filename`` at ``attrs.offset`` directly into
/// ``destination``.
///
/// The destination buffer must be writable for ``byte_count`` bytes when
/// ``byte_count`` is non-zero.
void DelayedInitializer::LoadBytesInto(const Attributes &attrs, uint8_t *destination,
                                       size_t byte_count) {
  if (byte_count == 0) {
    return;
  }
  EXT_ENFORCE_INVALID(destination != nullptr,
                      "kernel::DelayedInitializer received a null destination buffer.");

  const std::filesystem::path path(attrs.filename);
  EXT_ENFORCE_INVALID(std::filesystem::is_regular_file(path),
                      "kernel::DelayedInitializer filename is not a regular file: '",
                      attrs.filename, "'.");

  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  EXT_ENFORCE_INVALID(stream.good(), "kernel::DelayedInitializer unable to open file '",
                      attrs.filename, "'.");

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
  EXT_ENFORCE_INVALID(stream.tellg() == static_cast<std::streamoff>(attrs.offset),
                      "kernel::DelayedInitializer seek landed at an unexpected offset in '",
                      attrs.filename, "'.");
  stream.read(reinterpret_cast<char *>(destination), static_cast<std::streamsize>(byte_count));
  EXT_ENFORCE_INVALID(stream.gcount() == static_cast<std::streamsize>(byte_count),
                      "kernel::DelayedInitializer read failed for '", attrs.filename,
                      "': expected ", byte_count, " bytes, got ", stream.gcount(), ".");
}

void DelayedInitializer::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 0);
  RequireOutputCount(node, 1);
  const AttributeProto *shape_attr = FindAttribute(node, "shape");
  EXT_ENFORCE_INVALID(shape_attr != nullptr,
                      "RunNode: op 'DelayedInitializer' is missing 'shape' INTS attribute.");
  EXT_ENFORCE_INVALID(shape_attr->type() == AttributeProto::AttributeType::INTS,
                      "RunNode: attribute 'shape' of op 'DelayedInitializer' must be INTS.");
  onnx_kernels::kernel::DelayedInitializer::Attributes attrs;
  attrs.shape.reserve(shape_attr->ints().size());
  for (size_t i = 0; i < shape_attr->ints().size(); ++i) {
    attrs.shape.push_back(shape_attr->ints()[i]);
  }
  attrs.dtype = static_cast<int32_t>(GetRequiredAttributeInt(node, "dtype"));
  attrs.load_device = GetRequiredAttributeString(node, "load_device");
  attrs.runtime_device = GetRequiredAttributeString(node, "runtime_device");
  attrs.filename = GetRequiredAttributeString(node, "filename");
  attrs.offset = GetRequiredAttributeInt(node, "offset");
  onnx_kernels::kernel::DelayedInitializer kernel(rt.kernel_ctx(), std::move(attrs));
  SetOutput(node, 0, kernel(&rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
