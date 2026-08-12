// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file scoped_resource.h
 * @brief RAII helpers for OS-level resources and scope-exit guards.
 *
 * Provides three utilities:
 * - ScopedResource – a non-copyable RAII wrapper that calls a user-supplied
 *   @p Close function when the held value leaves scope.
 * - ScopedFd / ScopedHandle – ready-made specialisations for POSIX file
 *   descriptors and Win32 HANDLE values, respectively.
 * - ScopeExit – a non-copyable guard that invokes a callable when it is
 *   destroyed, regardless of how the enclosing scope is exited.
 */

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

namespace ONNX_LIGHT_NAMESPACE {

/**
 * @brief RAII wrapper that owns a single OS resource value.
 *
 * Holds a value of type `Traits::type` and calls `Traits::close` on it when the
 * wrapper is destroyed, unless ownership has been transferred via release().
 * Copying is disabled; move semantics are intentionally omitted to keep the
 * API minimal.
 *
 * The sentinel is obtained through `Traits::invalid()` rather than a non-type
 * template argument because some platform sentinels (e.g. Windows'
 * INVALID_HANDLE_VALUE, which reinterpret-casts -1 to a pointer) are not valid
 * converted constant expressions for a pointer-typed non-type template
 * parameter, even though permissive compilers accept them.
 *
 * @tparam Traits Type supplying `using type`, `static type invalid()` and
 *         `static void close(type)`. The destructor skips the `close` call
 *         when the held value compares equal to `invalid()`.
 */
template <typename Traits> class ScopedResource {
  using T = typename Traits::type;
  T val_;

public:
  /// Constructs a ScopedResource that takes ownership of @p v.
  /// @param v Resource value to manage; must compare equal to Traits::invalid() to be treated as
  /// empty.
  explicit ScopedResource(T v) : val_(v) {}

  /// Destroys the resource by calling Traits::close, unless the value equals Traits::invalid().
  ~ScopedResource() {
    if (val_ != Traits::invalid()) {
      Traits::close(val_);
    }
  }

  /// Returns the held resource value without releasing ownership.
  /// @returns The current resource value; equals Traits::invalid() when no resource is owned.
  T get() const { return val_; }

  /**
   * @brief Releases ownership and returns the held value.
   *
   * After the call the internal value is set to Traits::invalid() so the
   * destructor will not invoke Traits::close.
   *
   * @returns The resource value that was held before the call.
   */
  T release() {
    T tmp = val_;
    val_ = Traits::invalid();
    return tmp;
  }

  ScopedResource(const ScopedResource &) = delete;
  ScopedResource &operator=(const ScopedResource &) = delete;
};

#ifdef _WIN32
/// Traits describing how to close a Win32 HANDLE and its invalid sentinel.
struct HandleTraits {
  using type = HANDLE;
  static HANDLE invalid() { return INVALID_HANDLE_VALUE; }
  static void close(HANDLE h) { CloseHandle(h); }
};
/// RAII wrapper for a Win32 HANDLE; calls CloseHandle() on destruction.
using ScopedHandle = ScopedResource<HandleTraits>;
#endif

/// Traits describing how to close a file descriptor and its invalid sentinel.
struct FdTraits {
  using type = int;
  static constexpr int invalid() { return -1; }
  static void close(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    ::close(fd);
#endif
  }
};
/// RAII wrapper for a file descriptor; calls FdTraits::close() on destruction.
using ScopedFd = ScopedResource<FdTraits>;

/**
 * @brief Runs a callable unconditionally when the guard goes out of scope.
 *
 * The callable @p F must be noexcept-invocable; a @c static_assert enforces
 * this at instantiation time. Copying is disabled so that the guard fires
 * exactly once.
 *
 * @tparam F Callable type. Must satisfy `std::is_nothrow_invocable_v<F &>`.
 */
template <typename F> class ScopeExit {
  F fn_;

public:
  /// Constructs a ScopeExit that will invoke @p fn on destruction.
  /// @param fn Callable to invoke when the guard is destroyed; must be noexcept-invocable.
  explicit ScopeExit(F fn) : fn_(std::move(fn)) {}

  /// Invokes the stored callable.
  ~ScopeExit() noexcept {
    static_assert(std::is_nothrow_invocable_v<F &>, "ScopeExit callable must be noexcept");
    fn_();
  }

  ScopeExit(const ScopeExit &) = delete;
  ScopeExit &operator=(const ScopeExit &) = delete;
};

} // namespace ONNX_LIGHT_NAMESPACE
