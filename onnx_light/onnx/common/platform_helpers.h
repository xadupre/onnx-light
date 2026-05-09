// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_pb.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE {

inline bool is_processor_little_endian() {
  constexpr std::int32_t value = 1;
  return reinterpret_cast<const std::uint8_t *>(&value)[0] == 1;
}

} // namespace ONNX_LIGHT_NAMESPACE
