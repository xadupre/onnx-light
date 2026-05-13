// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/// @file onnx_pb.h
/// @brief Compatibility forwarding header for code that includes
///        <tt>onnx_pb.h</tt> relative to the <tt>onnx/</tt> include root.
///
/// The canonical location of this header in onnx-light is
/// ``onnx/common/onnx_pb.h``.  This stub ensures that any include of the
/// form ``#include "onnx_pb.h"`` or ``#include "onnx/onnx_pb.h"`` (as
/// produced by vendored ONNX sources that expect the standard layout) still
/// resolves correctly.

#pragma once

#include "onnx/common/onnx_pb.h"
