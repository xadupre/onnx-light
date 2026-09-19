// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_dlpack.h"
#include <nanobind/ndarray.h>

namespace ONNX_LIGHT_NAMESPACE {

/// Returns the shared native dtype mapping in nanobind's descriptor type.
inline nanobind::dlpack::dtype DLPackDtypeFromOnnx(int32_t data_type,
                                                   const char *producer = "Tensor") {
  const auto dtype = DLPackDataTypeFromOnnx(data_type, producer);
  return {dtype.code, dtype.bits, dtype.lanes};
}

} // namespace ONNX_LIGHT_NAMESPACE
