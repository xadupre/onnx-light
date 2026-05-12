// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Compatibility stub: onnx-light does not use protobuf.
// onnx/onnx-data.pb.h is included by some ONNX headers that have been vendored
// into onnx-light unchanged.  This stub redirects to the onnx-light equivalents.

#pragma once

#include "onnx/common/onnx_pb.h"

namespace ONNX_LIGHT_NAMESPACE {

// Mirrors the Version enum from the ONNX protobuf schema.
// IR_VERSION is the current IR version number understood by this library.
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
  IR_VERSION = 10,
};

} // namespace ONNX_LIGHT_NAMESPACE
