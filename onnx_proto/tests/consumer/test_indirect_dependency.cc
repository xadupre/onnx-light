// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_proto/onnx_dlpack.h"

int main() {
  const auto dtype =
      ONNX_LIGHT_NAMESPACE::DLPackDataTypeFromOnnx(ONNX_LIGHT_NAMESPACE::TensorProto::FLOAT);
  return dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1 ? 0 : 1;
}
