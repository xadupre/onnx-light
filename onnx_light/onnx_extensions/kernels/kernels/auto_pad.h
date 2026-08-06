// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#ifndef ONNX_LIGHT_NAMESPACE
#define ONNX_LIGHT_NAMESPACE onnx_light
#endif

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

/// Padding strategy for convolutional and pooling operators (ONNX
/// ``auto_pad`` attribute).
///
/// Values map directly to the ONNX attribute strings:
///  * ``kNotSet``   → ``"NOTSET"``  (default; use explicit ``pads``).
///  * ``kSameUpper`` → ``"SAME_UPPER"`` (output size = ceil(input / stride);
///    extra padding appended at the end of each spatial axis).
///  * ``kSameLower`` → ``"SAME_LOWER"`` (same output size rule; extra
///    padding prepended at the beginning of each spatial axis).
///  * ``kValid``    → ``"VALID"``   (no padding; output size =
///    floor((input − effective_kernel) / stride) + 1).
enum class AutoPad : uint8_t {
  kNotSet = 0,
  kSameUpper = 1,
  kSameLower = 2,
  kValid = 3,
};

/// Converts an ONNX ``auto_pad`` attribute string to an ``AutoPad``
/// enumerator. Unrecognised values (including the empty string) map to
/// ``AutoPad::kNotSet``.
inline AutoPad AutoPadFromString(const std::string &s) {
  if (s == "SAME_UPPER")
    return AutoPad::kSameUpper;
  if (s == "SAME_LOWER")
    return AutoPad::kSameLower;
  if (s == "VALID")
    return AutoPad::kValid;
  return AutoPad::kNotSet;
}

/// Returns the ONNX ``auto_pad`` attribute string for an ``AutoPad``
/// enumerator.
inline constexpr const char *AutoPadToString(AutoPad ap) {
  switch (ap) {
  case AutoPad::kSameUpper:
    return "SAME_UPPER";
  case AutoPad::kSameLower:
    return "SAME_LOWER";
  case AutoPad::kValid:
    return "VALID";
  default:
    return "NOTSET";
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
