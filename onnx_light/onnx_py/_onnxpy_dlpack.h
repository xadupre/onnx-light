// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx.h"
#include <nanobind/ndarray.h>

namespace ONNX_LIGHT_NAMESPACE {

/// Maps whole-byte ONNX element types to exact DLPack descriptors.
inline nanobind::dlpack::dtype DLPackDtypeFromOnnx(int32_t data_type,
                                                   const char *producer = "Tensor") {
  using Code = nanobind::dlpack::dtype_code;
  auto make = [](Code code, uint8_t bits) -> nanobind::dlpack::dtype {
    return {static_cast<uint8_t>(code), bits, 1};
  };
  switch (static_cast<TensorProto::DataType>(data_type)) {
  case TensorProto::FLOAT:
    return make(Code::Float, 32);
  case TensorProto::DOUBLE:
    return make(Code::Float, 64);
  case TensorProto::FLOAT16:
    return make(Code::Float, 16);
  case TensorProto::BFLOAT16:
    return make(Code::Bfloat, 16);
  case TensorProto::UINT8:
    return make(Code::UInt, 8);
  case TensorProto::INT8:
    return make(Code::Int, 8);
  case TensorProto::UINT16:
    return make(Code::UInt, 16);
  case TensorProto::INT16:
    return make(Code::Int, 16);
  case TensorProto::UINT32:
    return make(Code::UInt, 32);
  case TensorProto::INT32:
    return make(Code::Int, 32);
  case TensorProto::UINT64:
    return make(Code::UInt, 64);
  case TensorProto::INT64:
    return make(Code::Int, 64);
  case TensorProto::BOOL:
    return make(Code::Bool, 8);
  case TensorProto::COMPLEX64:
    return make(Code::Complex, 64);
  case TensorProto::COMPLEX128:
    return make(Code::Complex, 128);
  case TensorProto::FLOAT8E4M3FN:
    return make(Code::Float8_E4M3FN, 8);
  case TensorProto::FLOAT8E4M3FNUZ:
    return make(Code::Float8_E4M3FNUZ, 8);
  case TensorProto::FLOAT8E5M2:
    return make(Code::Float8_E5M2, 8);
  case TensorProto::FLOAT8E5M2FNUZ:
    return make(Code::Float8_E5M2FNUZ, 8);
  case TensorProto::FLOAT8E8M0:
    return make(Code::Float8_E8M0FNU, 8);
  default:
    EXT_THROW_INVALID(producer, ".__dlpack__: data type '",
                      TensorProto::DataType_Name(static_cast<TensorProto::DataType>(data_type)),
                      "' cannot be exported through DLPack (STRING and sub-byte packed types are "
                      "not supported).");
  }
}

} // namespace ONNX_LIGHT_NAMESPACE
