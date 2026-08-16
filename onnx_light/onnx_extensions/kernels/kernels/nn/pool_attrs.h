// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared spatial-pooling attribute parsing used by the ``Run`` methods of the
// pooling kernels (AveragePool, MaxPool, LpPool). Defined at ``onnx_kernels``
// scope so unqualified ``Shape`` resolves to the ``core::runtime`` fixed-rank
// shape rather than the like-named ``kernel::Shape`` kernel class.

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_extensions/kernels/kernels/auto_pad.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels {
using namespace ::onnx_light::core::runtime;

// Shared spatial pooling attributes consumed by AveragePool / MaxPool / LpPool
// (and other pool-style ops). The defaults mirror the ONNX schema.
struct PoolCommonAttrs {
  Shape kernel_shape;
  Shape strides;
  Shape pads;
  Shape dilations;
  bool ceil_mode;
  onnx_kernels::kernel::AutoPad auto_pad;
};

inline PoolCommonAttrs ParsePoolCommonAttrs(const NodeProto &node) {
  PoolCommonAttrs a;
  a.kernel_shape = GetAttributeShapeOrDefault(node, "kernel_shape", Shape{});
  a.strides = GetAttributeShapeOrDefault(node, "strides", Shape{});
  a.pads = GetAttributeShapeOrDefault(node, "pads", Shape{});
  a.dilations = GetAttributeShapeOrDefault(node, "dilations", Shape{});
  a.ceil_mode = GetAttributeIntOrDefault(node, "ceil_mode", 0) != 0;
  a.auto_pad = onnx_kernels::kernel::AutoPadFromString(
      GetAttributeStringOrDefault(node, "auto_pad", "NOTSET"));
  return a;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels
