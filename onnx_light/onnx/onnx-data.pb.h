// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/// @file onnx-data.pb.h
/// @brief Compatibility layer for the protobuf-generated
///        <tt>onnx-data.pb.h</tt> header from ONNX.
///
/// The reference ONNX C++ distribution generates this header from
/// ``onnx-data.proto`` and uses it as the canonical definition point for the
/// ``Version`` enum. Some ONNX headers vendored into *onnx-light* still include
/// ``onnx/onnx-data.pb.h`` unchanged.
///
/// *onnx-light* does not depend on protobuf code generation, so this header
/// provides a minimal API-compatible replacement:
/// - it includes ``onnx/common/onnx_pb.h`` for shared ONNX C++ symbols;
/// - it defines the ONNX IR ``Version`` enum expected by downstream code.

#pragma once

#include "onnx/common/onnx_pb.h"

namespace ONNX_LIGHT_NAMESPACE {

/// @brief ONNX IR version constants mirrored from the ONNX protobuf schema.
///
/// ``IR_VERSION`` is the latest IR version supported by this build of
/// *onnx-light*.
enum Version {
  START_VERSION = 0,
  IR_VERSION_2017_10_10 = 1,
  IR_VERSION_2017_10_30 = 2,
  IR_VERSION_2017_11_3 = 3,
  IR_VERSION_2019_1_22 = 4,
  IR_VERSION_2019_3_18 = 5,
  IR_VERSION_2019_9_19 = 6,
  IR_VERSION_2020_5_8 = 7,
  IR_VERSION_2021_7_30 = 8,
  IR_VERSION_2023_5_5 = 9,
  IR_VERSION_2024_3_25 = 10,
  IR_VERSION_2025_05_12 = 11,
  IR_VERSION_2025_08_26 = 12,
  IR_VERSION = 13,
};

} // namespace ONNX_LIGHT_NAMESPACE
