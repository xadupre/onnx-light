// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_pb.h"

#include <type_traits>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ONNX_NAMESPACE {

template <auto Invalid, void (*Close)(decltype(Invalid))> class ScopedResource {
  using T = decltype(Invalid);
  T val_;

public:
  explicit ScopedResource(T v) : val_(v) {}

  ~ScopedResource() {
    if (val_ != Invalid) {
      Close(val_);
    }
  }

  T get() const { return val_; }

  T release() {
    T tmp = val_;
    val_ = Invalid;
    return tmp;
  }

  ScopedResource(const ScopedResource &) = delete;
  ScopedResource &operator=(const ScopedResource &) = delete;
};

#ifdef _WIN32
inline void close_handle(HANDLE h) { CloseHandle(h); }
using ScopedHandle = ScopedResource<INVALID_HANDLE_VALUE, close_handle>;
#endif

inline void close_fd(int fd) {
#ifdef _WIN32
  _close(fd);
#else
  close(fd);
#endif
}
using ScopedFd = ScopedResource<-1, close_fd>;

template <typename F> class ScopeExit {
  F fn_;

public:
  explicit ScopeExit(F fn) : fn_(std::move(fn)) {}

  ~ScopeExit() noexcept {
    static_assert(std::is_nothrow_invocable_v<F &>, "ScopeExit callable must be noexcept");
    fn_();
  }

  ScopeExit(const ScopeExit &) = delete;
  ScopeExit &operator=(const ScopeExit &) = delete;
};

} // namespace ONNX_NAMESPACE
