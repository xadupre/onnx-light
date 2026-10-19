#pragma once

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#undef OPTIONAL

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace onnx_light_test {

/** Creates a directory junction without requiring symbolic-link privileges. */
inline std::error_code CreateWindowsJunction(const std::filesystem::path &junction,
                                             const std::filesystem::path &target,
                                             bool create_directory = true) {
  std::error_code error;
  const auto absolute_target = std::filesystem::absolute(target, error);
  if (error)
    return error;
  const std::wstring print_name = absolute_target.native();
  const std::wstring substitute_name =
      print_name.compare(0, 4, L"\\\\?\\") == 0
          ? L"\\??\\" + print_name.substr(4)
          : (print_name.compare(0, 2, L"\\\\") == 0 ? L"\\??\\UNC\\" + print_name.substr(2)
                                                    : L"\\??\\" + print_name);
  struct JunctionHeader {
    DWORD tag;
    WORD data_length;
    WORD reserved;
    WORD substitute_offset;
    WORD substitute_length;
    WORD print_offset;
    WORD print_length;
  };
  static_assert(sizeof(JunctionHeader) == 16);
  const size_t substitute_bytes = substitute_name.size() * sizeof(wchar_t);
  const size_t print_bytes = print_name.size() * sizeof(wchar_t);
  const size_t print_offset = substitute_bytes + sizeof(wchar_t);
  const size_t total = sizeof(JunctionHeader) + print_offset + print_bytes + sizeof(wchar_t);
  if (total > MAXIMUM_REPARSE_DATA_BUFFER_SIZE)
    return std::make_error_code(std::errc::filename_too_long);
  const JunctionHeader header{IO_REPARSE_TAG_MOUNT_POINT,
                              static_cast<WORD>(total - 8),
                              0,
                              0,
                              static_cast<WORD>(substitute_bytes),
                              static_cast<WORD>(print_offset),
                              static_cast<WORD>(print_bytes)};
  std::vector<uint8_t> buffer(total, 0);
  std::memcpy(buffer.data(), &header, sizeof(header));
  std::memcpy(buffer.data() + sizeof(header), substitute_name.data(), substitute_bytes);
  std::memcpy(buffer.data() + sizeof(header) + print_offset, print_name.data(), print_bytes);
  if (create_directory && !std::filesystem::create_directory(junction, error))
    return error ? error : std::make_error_code(std::errc::file_exists);
  HANDLE handle = ::CreateFileW(
      junction.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    error = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
  } else {
    DWORD returned = 0;
    if (!::DeviceIoControl(handle, FSCTL_SET_REPARSE_POINT, buffer.data(),
                           static_cast<DWORD>(buffer.size()), nullptr, 0, &returned, nullptr))
      error = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    ::CloseHandle(handle);
  }
  if (error && create_directory) {
    std::error_code ignored;
    std::filesystem::remove(junction, ignored);
  }
  return error;
}

} // namespace onnx_light_test

#endif
