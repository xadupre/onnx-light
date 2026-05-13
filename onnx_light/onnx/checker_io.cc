// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// checker_io.cc – minimal, self-contained implementation of the external-data
// filesystem-security functions declared in onnx/checker.h.
//
// These functions (resolve_external_data_location, open_external_data, and
// check_is_experimental_op) do not depend on shape inference or the rest of
// the ONNX checker logic.  Keeping them in a separate translation unit lets
// the library build without requiring a full port of the protobuf-heavy
// shape-inference machinery.

#include "onnx/checker.h"

#include <cstdint>
#include <filesystem> // NOLINT(build/c++17)
#include <string>
#include <unordered_set>

#include "onnx/common/path.h"
#include "onnx/common/scoped_resource.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// Kernel-level path containment: prefer openat2 (Linux 5.6+) or
// O_RESOLVE_BENEATH (FreeBSD 13+, macOS 15+) when available.
#ifdef __linux__
#include <sys/syscall.h>
#ifdef SYS_openat2
#define ONNX_HAS_OPENAT2
#if __has_include(<linux/openat2.h>)
#include <linux/openat2.h>
#else
struct open_how {
  uint64_t flags;
  uint64_t mode;
  uint64_t resolve;
};
#define RESOLVE_NO_SYMLINKS 0x04
#define RESOLVE_BENEATH 0x08
#endif
#endif // SYS_openat2
#endif // __linux__

#ifdef O_RESOLVE_BENEATH
#define ONNX_HAS_RESOLVE_BENEATH
#endif

#endif // _WIN32

namespace ONNX_LIGHT_NAMESPACE {
namespace checker {

using ONNX_LIGHT_NAMESPACE::ScopedFd;
using ONNX_LIGHT_NAMESPACE::utf8_to_path;
#ifdef _WIN32
using ONNX_LIGHT_NAMESPACE::ScopedHandle;
#endif

#ifdef _WIN32
// Compare two BY_HANDLE_FILE_INFORMATION structs by volume + file index (inode equivalent).
static bool same_file(const BY_HANDLE_FILE_INFORMATION &a, const BY_HANDLE_FILE_INFORMATION &b) {
  return a.dwVolumeSerialNumber == b.dwVolumeSerialNumber && a.nFileIndexHigh == b.nFileIndexHigh &&
         a.nFileIndexLow == b.nFileIndexLow;
}
#endif

// Canonicalize data_path, verify containment within base_dir.
static std::filesystem::path verify_path_containment(const std::filesystem::path &data_path,
                                                     const std::string &base_dir,
                                                     const std::string &tensor_name) {
  std::error_code ec;
  auto canonical_data = std::filesystem::weakly_canonical(data_path, ec);
  if (ec) {
    fail_check("Tensor ", tensor_name,
               " external data path could not be canonicalized: ", ec.message());
  }
  auto canonical_base = std::filesystem::weakly_canonical(utf8_to_path(base_dir), ec);
  if (ec) {
    fail_check("Tensor ", tensor_name,
               " base directory could not be canonicalized: ", ec.message());
  }
  auto base_str = canonical_base.native();
  if (!base_str.empty() && base_str.back() != std::filesystem::path::preferred_separator) {
    base_str += std::filesystem::path::preferred_separator;
  }
  if (canonical_data.native().find(base_str) != 0 && canonical_data != canonical_base) {
    // NOSONAR: prefix-match for path containment; exact equality is the
    // base-dir-is-the-data-file edge case (allowed).  std::filesystem::path
    // provides no starts_with() in C++17.
    fail_check("Tensor ", tensor_name, " external data resolves outside model directory.");
  }
  return canonical_data;
}

std::filesystem::path resolve_external_data_location(const std::string &base_dir,
                                                     const std::string &location,
                                                     const std::string &tensor_name) {
  auto base_dir_path = utf8_to_path(base_dir);
  auto file_path = utf8_to_path(location);
  if (file_path.empty()) {
    fail_check("Location of external TensorProto ( tensor name: ", tensor_name,
               ") should not be empty.");
  }
  if (file_path.is_absolute()) {
    fail_check("Location of external TensorProto ( tensor name: ", tensor_name,
               ") should be a relative path, but it is an absolute path: ", location);
  }
  auto relative_path = file_path.lexically_normal().make_preferred();
  if (relative_path.native().find(std::filesystem::path("..").native()) !=
      std::filesystem::path::string_type::npos) {
    fail_check("Data of TensorProto ( tensor name: ", tensor_name, ") should be file inside '",
               base_dir, "', but '", location, "' points outside the directory.");
  }
  {
    // External data location must be a plain filename with no directory component.
    auto parent = relative_path.parent_path();
    if (!parent.empty() && parent != std::filesystem::path(".")) {
      fail_check("Location of external TensorProto ( tensor name: ", tensor_name,
                 ") must be a filename with no folder, but got: ", location);
    }
  }
  auto data_path = base_dir_path / relative_path;
  auto data_path_str = path_to_utf8(data_path);
  // Do not allow symlinks or directories.
  if (data_path.empty() || std::filesystem::is_symlink(data_path)) {
    fail_check("Data of TensorProto ( tensor name: ", tensor_name, ") should be stored in ",
               data_path_str, ", but it is a symbolic link.");
  }
  // Verify canonical containment (catches parent-dir symlinks).
  if (data_path_str[0] != '#') {
    verify_path_containment(data_path, base_dir, tensor_name);
  }
  if (data_path_str[0] != '#' && !std::filesystem::is_regular_file(data_path)) {
    fail_check("Data of TensorProto ( tensor name: ", tensor_name, ") should be stored in ",
               data_path_str, ", but it is not regular file.");
  }
  // Do not allow hardlinks, as they can be used to read arbitrary files.
  if (data_path_str[0] != '#' && std::filesystem::hard_link_count(data_path) > 1) {
    fail_check("Data of TensorProto ( tensor name: ", tensor_name, ") should be stored in ",
               data_path_str,
               ", but it has multiple hard links, indicating a potential hardlink attack.");
  }
  return data_path;
}

static std::filesystem::path validate_write_location(const std::string &base_dir,
                                                     const std::string &location,
                                                     const std::string &tensor_name) {
  auto file_path = utf8_to_path(location);
  if (file_path.empty() || file_path.is_absolute()) {
    fail_check("External data location for tensor ", tensor_name,
               " is empty or absolute: ", location);
  }
  auto rel = file_path.lexically_normal().make_preferred();
  if (rel == std::filesystem::path(".")) {
    fail_check("External data location for tensor ", tensor_name, " is invalid: ", location);
  }
  if (rel.native().find(std::filesystem::path("..").native()) !=
      std::filesystem::path::string_type::npos) {
    // NOSONAR: std::filesystem::path::string_type (std::string / std::wstring)
    // has no contains() in C++17; find() returning npos is the standard idiom.
    fail_check("External data location for tensor ", tensor_name, " contains '..': ", location);
  }
  {
    // External data location must be a plain filename with no directory component.
    auto parent = rel.parent_path();
    if (!parent.empty() && parent != std::filesystem::path(".")) {
      fail_check("External data location for tensor ", tensor_name,
                 " must be a filename with no folder, but got: ", location);
    }
  }
  return utf8_to_path(base_dir) / rel;
}

#ifdef _WIN32

int64_t open_external_data(const std::string &base_dir, const std::string &location,
                           const std::string &tensor_name, bool read_only) {
  std::filesystem::path data_path;
  if (read_only) {
    data_path = resolve_external_data_location(base_dir, location, tensor_name);
  } else {
    data_path = validate_write_location(base_dir, location, tensor_name);
    // Pre-open parent-dir check (final-component-only flags don't protect parents).
    verify_path_containment(data_path.parent_path(), base_dir, tensor_name);
  }

  // CreateFileW with FILE_FLAG_OPEN_REPARSE_POINT: opens reparse point itself
  // (symlink/junction) without following, and returns CRT-independent HANDLE.
  DWORD access = read_only ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
  DWORD creation = read_only ? OPEN_EXISTING : OPEN_ALWAYS;
  HANDLE h = CreateFileW(data_path.native().c_str(), access, FILE_SHARE_READ, nullptr, creation,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    fail_check("Cannot open external data for tensor ", tensor_name);
  }
  ScopedHandle guard(h);

  // Reject symlinks/junctions.
  BY_HANDLE_FILE_INFORMATION file_info;
  if (!GetFileInformationByHandle(h, &file_info)) {
    fail_check("Tensor ", tensor_name, " external data: cannot query file information.");
  }
  if (file_info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
    fail_check("Tensor ", tensor_name, " external data is a reparse point (symlink/junction).");
  }

  // Inode comparison: open canonical path, compare volume + file index.
  auto canonical_data = verify_path_containment(data_path, base_dir, tensor_name);
  HANDLE h2 = CreateFileW(canonical_data.native().c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE,
                          nullptr, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (h2 == INVALID_HANDLE_VALUE) {
    fail_check("Tensor ", tensor_name,
               " external data: cannot open canonical path for verification.");
  }
  ScopedHandle guard2(h2);
  BY_HANDLE_FILE_INFORMATION canon_info;
  if (!GetFileInformationByHandle(h2, &canon_info)) {
    fail_check("Tensor ", tensor_name,
               " external data: cannot query canonical path file information.");
  }
  if (!same_file(file_info, canon_info)) {
    fail_check("Tensor ", tensor_name,
               " external data: fd/path mismatch (possible TOCTOU attack).");
  }

  // Hardlink check (fail closed).
  if (file_info.nNumberOfLinks > 1) {
    fail_check("Tensor ", tensor_name,
               " external data has multiple hard links (possible hardlink attack).");
  }

  // Convert HANDLE -> CRT fd so callers get a uniform fd on all platforms.
  HANDLE h_released = guard.release();
  int flags = read_only ? (_O_RDONLY | _O_BINARY) : (_O_RDWR | _O_BINARY);
  int fd = _open_osfhandle(reinterpret_cast<intptr_t>(h_released), flags);
  if (fd < 0) {
    CloseHandle(h_released);
    fail_check("Cannot convert handle to fd for tensor ", tensor_name);
  }
  return static_cast<int64_t>(fd);
}

#else // POSIX

// Try kernel-level contained open.
// Returns fd >= 0 on success, -1 if feature unavailable (fall through to legacy).
// Calls fail_check on kernel rejection (symlink, escape) — does NOT fall through.
static int try_kernel_contained_open(const std::string &base_dir, const std::string &location,
                                     [[maybe_unused]] const std::string &tensor_name,
                                     [[maybe_unused]] bool read_only) {
  int raw_dirfd = open(base_dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (raw_dirfd < 0) {
    return -1;
  }
  ScopedFd dirfd_guard(raw_dirfd);
  auto rel = std::filesystem::path(location).lexically_normal().string();
  int fd = -1;

#ifdef ONNX_HAS_OPENAT2
  static thread_local bool openat2_supported = true;
  if (openat2_supported) {
    struct open_how how = {};
    how.flags = read_only ? static_cast<uint64_t>(O_RDONLY | O_CLOEXEC)
                          : static_cast<uint64_t>(O_CREAT | O_RDWR | O_CLOEXEC);
    how.mode = read_only ? 0 : 0600;
    how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS;
    fd = static_cast<int>(syscall(SYS_openat2, raw_dirfd, rel.c_str(), &how, sizeof(how)));
    if (fd < 0) {
      if (errno == ENOSYS) {
        openat2_supported = false; // kernel too old, fall through
      } else {
        fail_check("Cannot open external data for tensor ", tensor_name, " (kernel rejected path)");
      }
    }
  }
#elif defined(ONNX_HAS_RESOLVE_BENEATH)
  int flags = O_RESOLVE_BENEATH | O_CLOEXEC;
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  if (read_only) {
    fd = openat(raw_dirfd, rel.c_str(), flags | O_RDONLY);
  } else {
    fd = openat(raw_dirfd, rel.c_str(), flags | O_CREAT | O_RDWR, 0600);
  }
  if (fd < 0) {
    fail_check("Cannot open external data for tensor ", tensor_name);
  }
#endif

  return fd;
}

int64_t open_external_data(const std::string &base_dir, const std::string &location,
                           const std::string &tensor_name, bool read_only) {
  // Pre-open validation (defense-in-depth).
  std::filesystem::path data_path;
  if (read_only) {
    data_path = resolve_external_data_location(base_dir, location, tensor_name);
  } else {
    data_path = validate_write_location(base_dir, location, tensor_name);
    // Pre-open parent-dir check (final-component-only flags don't protect parents).
    verify_path_containment(data_path.parent_path(), base_dir, tensor_name);
  }

  // Try kernel-level contained open (openat2 or O_RESOLVE_BENEATH).
  int fd = try_kernel_contained_open(base_dir, location, tensor_name, read_only);
  bool kernel_verified = (fd >= 0);

  // Fallback: open() + O_NOFOLLOW + post-open inode verification.
  if (fd < 0) {
    int flags = read_only ? O_RDONLY : (O_CREAT | O_RDWR);
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    fd = read_only ? open(data_path.c_str(), flags) : open(data_path.c_str(), flags, 0600);
    if (fd == -1) {
      fail_check("Cannot open external data for tensor ", tensor_name);
    }
  }
  ScopedFd guard(fd);

  // Post-open checks (fail closed).
  struct stat fd_stat {};
  if (fstat(fd, &fd_stat) != 0) {
    fail_check("Tensor ", tensor_name, " external data: fstat failed.");
  }
  if (!kernel_verified) {
    // Verify containment via canonical path + inode comparison.
    auto canonical_data = verify_path_containment(data_path, base_dir, tensor_name);
    struct stat path_stat {};
    if (stat(canonical_data.c_str(), &path_stat) != 0) {
      fail_check("Tensor ", tensor_name, " external data: cannot stat canonical path.");
    }
    if (path_stat.st_dev != fd_stat.st_dev || path_stat.st_ino != fd_stat.st_ino) {
      fail_check("Tensor ", tensor_name,
                 " external data: fd/path mismatch (possible TOCTOU attack).");
    }
  }
  if (fd_stat.st_nlink > 1) {
    fail_check("Tensor ", tensor_name,
               " external data has multiple hard links (possible hardlink attack).");
  }

  return guard.release();
}

#endif

static const std::unordered_set<std::string> experimental_ops = {"ATen",
                                                                 "Affine",
                                                                 "ConstantFill",
                                                                 "Crop",
                                                                 "DynamicSlice",
                                                                 "GRUUnit",
                                                                 "GivenTensorFill",
                                                                 "ImageScaler",
                                                                 "ParametricSoftplus",
                                                                 "Scale",
                                                                 "ScaledTanh"};

bool check_is_experimental_op(const NodeProto &node) {
  const std::string domain = node.ref_domain().as_string();
  const std::string op_type = node.ref_op_type().as_string();
  return (domain == ONNX_DOMAIN || domain == "ai.onnx") && experimental_ops.count(op_type) != 0;
}

} // namespace checker
} // namespace ONNX_LIGHT_NAMESPACE
