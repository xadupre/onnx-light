// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/raw_buffer_allocator.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {
namespace detail {

// Holds temporary typed storage for kernel helpers. Uses the provided
// allocator when available and falls back to std::vector otherwise.
template <typename T> struct TemporaryTypedBuffer {
  std::vector<T> fallback;
  RawBufferAllocator *allocator = nullptr;
  RawBuffer *buffer = nullptr;
  std::size_t size = 0;

  TemporaryTypedBuffer() = default;

  TemporaryTypedBuffer(std::size_t count, RawBufferAllocator *buffer_allocator, const char *name)
      : size(count) {
    if (buffer_allocator != nullptr) {
      RawBuffer *allocated = buffer_allocator->Allocate(count * sizeof(T));
      EXT_ENFORCE_INVALID(allocated != nullptr, name, " allocator returned null.");
      if (allocated->size() < count * sizeof(T)) {
        buffer_allocator->Free(allocated);
        EXT_THROW_INVALID(name, " allocator returned too small a buffer.");
      }
      if (reinterpret_cast<std::uintptr_t>(allocated->data()) % alignof(T) != 0) {
        buffer_allocator->Free(allocated);
        EXT_THROW_INVALID(name, " allocator returned a misaligned buffer.");
      }
      allocator = buffer_allocator;
      buffer = allocated;
      return;
    }
    fallback.resize(count);
  }

  TemporaryTypedBuffer(const TemporaryTypedBuffer &) = delete;
  TemporaryTypedBuffer &operator=(const TemporaryTypedBuffer &) = delete;

  ~TemporaryTypedBuffer() {
    if (buffer != nullptr && allocator != nullptr) {
      allocator->Free(buffer);
    }
  }

  T *data() {
    EXT_ENFORCE_INVALID(buffer != nullptr || !fallback.empty() || size == 0,
                        "temporary buffer not allocated.");
    if (buffer != nullptr) {
      return reinterpret_cast<T *>(buffer->data());
    }
    return fallback.data();
  }

  void CopyFromBytes(const std::uint8_t *bytes) { std::memcpy(data(), bytes, size * sizeof(T)); }

  void ZeroFill() { std::memset(data(), 0, size * sizeof(T)); }
};

} // namespace detail
} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
