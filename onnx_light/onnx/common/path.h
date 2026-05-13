// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file path.h
 * @brief UTF-8 / filesystem path conversion utilities.
 *
 * Provides cross-platform helpers for converting between UTF-8 encoded
 * @c std::string values and @c std::filesystem::path objects.  On Windows,
 * two additional helpers convert between UTF-8 @c std::string and
 * @c std::wstring using the Win32 MultiByteToWideChar / WideCharToMultiByte
 * APIs.
 */

#pragma once

#include "common.h"

#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ONNX_LIGHT_NAMESPACE {

#ifdef _WIN32
/**
 * @brief Converts a UTF-8 encoded string to a wide string (Windows only).
 *
 * Uses the Win32 @c MultiByteToWideChar API with the @c CP_UTF8 code page
 * and the @c MB_ERR_INVALID_CHARS | @c MB_PRECOMPOSED flags so that invalid
 * UTF-8 sequences are rejected rather than silently replaced.
 *
 * @param utf8str UTF-8 encoded input string.
 * @return Corresponding @c std::wstring.
 * @throws std::runtime_error Throws when @c MultiByteToWideChar fails (e.g.
 *         the input contains invalid UTF-8 sequences).
 */
inline std::wstring utf8str_to_wstring(const std::string &utf8str) {
  if (utf8str.empty()) {
    return std::wstring();
  }
  int len = static_cast<int>(utf8str.size());
  auto size_required = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS | MB_PRECOMPOSED,
                                           utf8str.data(), len, nullptr, 0);
  if (size_required == 0) {
    auto last_error = GetLastError();
    ONNX_THROW("MultiByteToWideChar in utf8str_to_wstring returned error: ", last_error);
  }
  std::wstring ws_str(size_required, 0);
  auto converted_size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS | MB_PRECOMPOSED,
                                            utf8str.data(), len, &ws_str[0], size_required);
  if (converted_size == 0) {
    auto last_error = GetLastError();
    ONNX_THROW("MultiByteToWideChar in utf8str_to_wstring returned error: ", last_error);
  }
  return ws_str;
}

/**
 * @brief Converts a wide string to a UTF-8 encoded string (Windows only).
 *
 * Uses the Win32 @c WideCharToMultiByte API with the @c CP_UTF8 code page
 * and the @c WC_ERR_INVALID_CHARS flag so that invalid wide-character
 * sequences are rejected rather than silently replaced.
 *
 * @param ws_str Wide string input.
 * @return Corresponding UTF-8 encoded @c std::string.
 * @throws std::runtime_error Throws when @c WideCharToMultiByte fails (e.g.
 *         the input contains invalid wide-character sequences).
 */
inline std::string wstring_to_utf8str(const std::wstring &ws_str) {
  if (ws_str.empty()) {
    return std::string();
  }
  int len = static_cast<int>(ws_str.size());
  auto size_required = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, ws_str.data(), len,
                                           nullptr, 0, nullptr, nullptr);
  if (size_required == 0) {
    auto last_error = GetLastError();
    ONNX_THROW("WideCharToMultiByte in wstring_to_utf8str returned error: ", last_error);
  }
  std::string utf8str(size_required, 0);
  auto converted_size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, ws_str.data(), len,
                                            &utf8str[0], size_required, nullptr, nullptr);
  if (converted_size == 0) {
    auto last_error = GetLastError();
    ONNX_THROW("WideCharToMultiByte in wstring_to_utf8str returned error: ", last_error);
  }
  return utf8str;
}
#endif

/**
 * @brief Converts a UTF-8 encoded string to a @c std::filesystem::path.
 *
 * On Windows the string is first widened via utf8str_to_wstring so that
 * non-ASCII characters in the path are handled correctly.  On all other
 * platforms the path is constructed directly from the UTF-8 string.
 *
 * @param utf8 UTF-8 encoded path string.
 * @return Corresponding @c std::filesystem::path.
 */
inline std::filesystem::path utf8_to_path(const std::string &utf8) {
#ifdef _WIN32
  return std::filesystem::path(utf8str_to_wstring(utf8));
#else
  return std::filesystem::path(utf8);
#endif
}

/**
 * @brief Converts a @c std::filesystem::path to a UTF-8 encoded string.
 *
 * On Windows the path's native wide string is converted to UTF-8 via
 * wstring_to_utf8str.  On all other platforms the path's native string
 * representation (already UTF-8) is returned directly.
 *
 * @param p Filesystem path to convert.
 * @return UTF-8 encoded string representation of @p p.
 */
inline std::string path_to_utf8(const std::filesystem::path &p) {
#ifdef _WIN32
  return wstring_to_utf8str(p.wstring());
#else
  return p.string();
#endif
}

} // namespace ONNX_LIGHT_NAMESPACE
