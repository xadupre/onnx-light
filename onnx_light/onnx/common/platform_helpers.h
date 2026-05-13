// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file platform_helpers.h
 * @brief Platform-detection helper utilities.
 *
 * Provides lightweight, header-only helpers for querying properties of the
 * host platform at run time, such as the processor byte order.
 */

#pragma once

#include "onnx_pb.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {

/**
 * @brief Returns @c true when the host processor uses little-endian byte order.
 *
 * The check is performed at run time by writing a known 32-bit value into
 * memory and inspecting its first byte.  On a little-endian machine the least
 * significant byte comes first, so the first byte equals @c 1.
 *
 * @return @c true if the processor is little-endian; @c false otherwise.
 */
inline bool is_processor_little_endian() {
  constexpr std::int32_t value = 1;
  return reinterpret_cast<const std::uint8_t *>(&value)[0] == 1;
}

} // namespace ONNX_LIGHT_NAMESPACE
